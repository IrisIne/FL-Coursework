import torch
import torchvision
from torchvision.models import mobilenet_v3_large
import os
import shutil

import ray
from ray import serve
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse

import time
import math
import numpy as np
from sklearn.mixture import GaussianMixture
from collections import defaultdict

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class FLServer:
    def __init__(self):
        if os.path.exists("global_model.pth"):
            self.model = mobilenet_v3_large()
            self.model.classifier[3] = torch.nn.Linear(1280, 5)
            self.model.load_state_dict(torch.load("global_model.pth", map_location="cpu"))
        else:
            self.model = mobilenet_v3_large(weights="IMAGENET1K_V1")
            self.model.classifier[3] = torch.nn.Linear(1280, 5)
            torch.save(self.model.state_dict(), "global_model.pth")

        self.last_update_time = time.time()
        self.tau = 500

        self.client_updates = {}
        self.client_losses = {}
        self.num_clusters = 3
        self.cluster_assignments = {}
        self.iteration_counter = 0

    @app.get("/get_model")
    def get_model(self):
        return FileResponse("global_model.pth", media_type="application/octet-stream", filename="global_model.pth")

    def cluster_clients(self):
        if len(self.client_updates) < self.num_clusters:
            print("Not enough data for clustering.")
            return {}

        client_ids, updates, losses = [], [], []

        for client_id, model_update in self.client_updates.items():
            update_vector = torch.cat([param.flatten() for param in model_update.values()]).cpu().numpy()
            client_ids.append(client_id)
            updates.append(update_vector)
            losses.append(self.client_losses.get(client_id, 0.5))

        updates = np.array(updates)
        losses = np.array(losses).reshape(-1, 1)
        losses = (losses - np.mean(losses)) / np.std(losses)
        features = np.hstack((updates, losses))

        gmm = GaussianMixture(n_components=self.num_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(features)

        self.cluster_assignments = {client_ids[i]: cluster_labels[i] for i in range(len(client_ids))}
        print(f"Clustered Clients: {self.cluster_assignments}")

    def aggregate_updates_by_cluster(self):
        cluster_updates = defaultdict(list)

        for client_id, cluster in self.cluster_assignments.items():
            cluster_updates[cluster].append(self.client_updates[client_id])

        aggregated_cluster_updates = {}
        for cluster, updates in cluster_updates.items():
            mean_update = {k: torch.zeros_like(v) for k, v in updates[0].items()}

            for update in updates:
                for k in update:
                    mean_update[k] += update[k] / len(updates)

            aggregated_cluster_updates[cluster] = mean_update

        return aggregated_cluster_updates

    @app.post("/update_model")
    async def update_model(self, file: UploadFile = File(...)):
        current_time = time.time()
        delay = current_time - self.last_update_time

        alpha = math.exp(-delay / self.tau)
        print(f"Client update delay: {delay:.2f}s, computed alpha: {alpha:.4f}")

        client_id = file.filename.split("_")[1]
        with open("client_weights_v2.pth", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        client_weights = torch.load("client_weights_v2.pth", map_location="cpu")
        self.client_updates[client_id] = client_weights
        self.iteration_counter += 1

        if self.iteration_counter >= self.num_clusters:
            self.cluster_clients()
            self.iteration_counter = 0

        aggregated_updates = self.aggregate_updates_by_cluster()

        with torch.no_grad():
            for cluster, update in aggregated_updates.items():
                for param, client_param in zip(self.model.parameters(), update.values()):
                    param.data = (1 - alpha) * param.data + alpha * client_param.data

        torch.save(self.model.state_dict(), "global_model.pth")
        self.last_update_time = current_time

        return {"message": "Model updated successfully"}

    @app.post("/send_loss")
    async def send_loss(self, client_id: str = Form(...), loss_value: float = Form(...)):
        self.client_losses[client_id] = loss_value
        print(f"Received loss {loss_value:.4f} from client {client_id}")

        return {"message": "Loss received successfully"}


if __name__ == "__main__":
    ray.init(address="auto", ignore_reinit_error=True)
    serve.start(detached=True)
    serve.run(FLServer.bind(), route_prefix="/")
