import torch
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large

import os
import random
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import numpy as np
import collections
import copy
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


NUM_CLIENTS = 7
NUM_EPOCHS = 10

cifar100_metaclasses = {
    "aquatic_mammals": [4, 30, 55, 72, 95],
    "flowers": [6, 7, 14, 18, 24],
    "fruit_and_vegetables": [3, 42, 43, 88, 97],
    "household_furniture": [5, 20, 25, 84, 94],
    "vehicles_1": [8, 16, 53, 73, 91]
}

selected_metaclasses = list(cifar100_metaclasses.keys())
selected_classes = sum([cifar100_metaclasses[mc] for mc in selected_metaclasses], [])

class_mapping = {}
for meta_idx, (meta_name, class_list) in enumerate(cifar100_metaclasses.items()):
    for class_id in class_list:
        class_mapping[class_id] = meta_idx


def get_global_model():
    url = "http://fl-master:8000/get_model"
    response = requests.get(url)

    if response.status_code == 200:
        with open("global_model.pth", "wb") as f:
            f.write(response.content)

        model = torchvision.models.mobilenet_v3_large()
        model.classifier[3] = torch.nn.Linear(1280, len(cifar100_metaclasses))
        model.load_state_dict(torch.load("global_model.pth", map_location="cpu"))

        return model

    return None


def send_loss(client_id, loss_value):
    url = "http://fl-master:8000/send_loss"
    response = requests.post(url, data={'client_id': client_id, 'loss_value': loss_value})

    if response.status_code == 200:
        print(f"Client {client_id}: Loss value {loss_value} sent successfully.")
    else:
        print(f"Client {client_id}: Failed to send loss value.")


class CustomSubset(Dataset):
    def __init__(self, dataset, indices, class_mapping):
        self.dataset = dataset
        self.indices = indices
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, target = self.dataset[original_idx]
        target = self.class_mapping[target]

        return image, target


def split_dataset(full_dataset, client_id, stochastic=False):
    num_samples = len(full_dataset) // NUM_CLIENTS
    random.seed(client_id)

    if stochastic:
        subset_indices = random.sample(range(len(full_dataset)), num_samples)
    else:
        start_idx, end_idx = client_id * num_samples, min((client_id + 1) * num_samples, len(full_dataset))

        subset_indices = list(range(start_idx, end_idx))

    return torch.utils.data.Subset(full_dataset, subset_indices)


def get_hybrid_dataset(full_dataset, client_id):
    fixed_dataset = split_dataset(full_dataset, client_id, stochastic=False)

    def update_stochastic_subset():
        return split_dataset(full_dataset, client_id, stochastic=True)

    return fixed_dataset, update_stochastic_subset


def random_split_local(full_dataset, fixed_dataset, update_stochastic_subset, fixed_data_ratio):
    stochastic_dataset = update_stochastic_subset()
    hybrid_indices = list(fixed_dataset.indices[:int(fixed_data_ratio * len(fixed_dataset))]) + list(stochastic_dataset.indices[int(fixed_data_ratio * len(stochastic_dataset)):])

    local_dataset = torch.utils.data.Subset(full_dataset, hybrid_indices)

    train_size = int(0.75 * len(local_dataset))
    val_size = int(0.1 * len(local_dataset))
    test_size = len(local_dataset) - train_size - val_size
    train_dataset, val_dataset, local_test_dataset = torch.utils.data.random_split(local_dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, local_test_dataset


def train_model(trial):
    device = torch.device("cpu")
    model = get_global_model()

    transform_train = transforms.Compose([
        transforms.ToTensor(),

        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_indices = [i for i in range(len(full_dataset)) if full_dataset.targets[i] in selected_classes]
    test_indices = [i for i in range(len(test_dataset)) if test_dataset.targets[i] in selected_classes]

    trainset_filtered = CustomSubset(full_dataset, train_indices, class_mapping)
    testset_filtered = CustomSubset(test_dataset, test_indices, class_mapping)

    client_id = int(os.environ.get("CLIENT_ID", random.randint(0, NUM_CLIENTS - 1)))

    fixed_dataset, update_stochastic_subset = get_hybrid_dataset(trainset_filtered, client_id)

    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    lr = trial.suggest_float("lr", 0.0009, 0.0010, log=True)
    weight_decay = trial.suggest_categorical("weight_decay", [1e-6, 2e-6])
    mu = trial.suggest_float("mu", 3e-6, 7e-5, log=True)
    fixed_data_ratio = trial.suggest_float("fixed_data_ratio", 0.2, 0.5, step=0.1)

    train_dataset, val_dataset, local_test_dataset = random_split_local(trainset_filtered, fixed_dataset, update_stochastic_subset, fixed_data_ratio)

    print(f"Trial {trial.number}, Client {client_id}: batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}, Mu {mu}, fixed_data_ratio={fixed_data_ratio}")

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(local_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    global_model = copy.deepcopy(model)

    train_loss_history, train_acc_history = [], []
    val_loss_history, val_acc_history = [], []

    for epoch in range(NUM_EPOCHS):

        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            fedprox_loss = sum(((param - global_param) ** 2).sum() for param, global_param in zip(model.parameters(), global_model.parameters()))
            loss += (mu / 2) * fedprox_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(trainloader)
        train_acc = correct / total
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)


                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(valloader)
        val_acc = val_correct / val_total
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f"Client {client_id}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    model_path = f"client_{client_id}_weights.pth"

    torch.save(model.state_dict(), model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File {model_path} not found after saving.")

    files = {"file": open(model_path, "rb")}

    requests.post("http://fl-master:8000/update_model", files=files)

    send_loss(client_id, val_loss)

    lr_str = f"{lr:.5f}"
    wd_str = f"{weight_decay:.0e}"

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Train")
    plt.plot(val_loss_history, label="Validation", linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss (Client {client_id}, Trial {trial.number}, lr={lr_str}, wd={wd_str})")
    plt.legend()
    plt.savefig(f"loss_plot_client_{client_id}_trial_{trial.number}.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_history, label="Train")
    plt.plot(val_acc_history, label="Validation", linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy (Client {client_id}, Trial {trial.number}, lr={lr_str}, wd={wd_str})")
    plt.legend()
    plt.savefig(f"accuracy_plot_client_{client_id}_trial_{trial.number}.png")
    plt.close()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix (Client {client_id}, Trial {trial.number})")
    plt.savefig(f"conf_matrix_client_{client_id}_trial_{trial.number}.png")
    plt.close()

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    print(f"Client {client_id} Test Accuracy: {test_acc:.4f}, Test F1-score: {test_f1:.4f}")

    return val_acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(train_model, n_trials=3)

    print("Best params:", study.best_params)
