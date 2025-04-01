import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from torch.utils.data import DataLoader, DistributedSampler, random_split, Subset, Dataset

import os
import matplotlib.pyplot as plt

BATCH_SIZE = 32
NUM_EPOCHS = 30


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'node-master'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


class MobileNetV3Modified(nn.Module):
    def __init__(self):
        super(MobileNetV3Modified, self).__init__()

        self.model = mobilenet_v3_large(weights="IMAGENET1K_V1")
        self.model.classifier[3] = nn.Linear(1280, 5)

    def forward(self, x):
        return self.model(x)


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


def train(rank, world_size):
    setup(rank, world_size)
    device = torch.device("cpu")

    cifar100_metaclasses_5 = {
        "aquatic_mammals": [4, 30, 55, 72, 95],
        "flowers": [54, 62, 70, 82, 92],
        "fruit_and_vegetables": [0, 51, 53, 57, 83],
        "household_furniture": [5, 20, 25, 84, 94],
        "vehicles_1": [8, 13, 48, 58, 90],
    }

    selected_metaclasses = list(cifar100_metaclasses_5.keys())
    selected_classes = sum([cifar100_metaclasses_5[mc] for mc in selected_metaclasses], [])

    class_mapping = {}
    for meta_idx, (meta_name, class_list) in enumerate(cifar100_metaclasses_5.items()):
        for class_id in class_list:
            class_mapping[class_id] = meta_idx

    transform_train = transforms.Compose([
        transforms.ToTensor(),

        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((224, 224)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)

    train_indices = [i for i in range(len(full_trainset)) if full_trainset.targets[i] in selected_classes]

    trainset_filtered = CustomSubset(full_trainset, train_indices, class_mapping)

    train_size = int(0.9 * len(trainset_filtered))
    val_size = len(trainset_filtered) - train_size
    trainset, valset = random_split(trainset_filtered, [train_size, val_size])

    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train set size: {len(trainset)}")
    print(f"Validation set size: {len(valset)}")

    model = MobileNetV3Modified()
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-06)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss, correct, total = 0.0, 0, 0
        train_sampler.set_epoch(epoch)

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(trainloader)
        train_acc = correct / total

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

        if epoch % 3 == 0:
            train_loss_tensor = torch.tensor(train_loss).to(device)
            train_acc_tensor = torch.tensor(train_acc).to(device)
            val_loss_tensor = torch.tensor(val_loss).to(device)
            val_acc_tensor = torch.tensor(val_acc).to(device)

            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)

            train_loss = train_loss_tensor.item() / world_size
            train_acc = train_acc_tensor.item() / world_size
            val_loss = val_loss_tensor.item() / world_size
            val_acc = val_acc_tensor.item() / world_size

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        scheduler.step()

        print(f"Rank {rank}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if rank == 0:
        torch.save(model.module.state_dict(), "mobilenetv3_cifar100.pth")

        print("Model is saved: mobilenetv3_cifar100.pth")

        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label="Train")
        plt.plot(val_loss_history, label="Validation", linestyle ="dashed")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Train/Validation Loss")
        plt.savefig("loss_plot.png")

        plt.figure(figsize=(10, 5))
        plt.plot(train_acc_history, label="Train")
        plt.plot(val_acc_history, label="Validation", linestyle="dashed")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Train/Validation Accuracy")
        plt.savefig("accuracy_plot.png")

        print("Graphs saved: loss_plot.png and accuracy_plot.png")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 3
    rank = int(os.environ['RANK'])
    train(rank, world_size)
