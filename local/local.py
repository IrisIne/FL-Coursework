import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import mobilenet_v2
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader, Subset, Dataset

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


if not os.path.exists('plots'):
    os.makedirs('plots')


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

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
full_testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

train_indices = [i for i in range(len(full_trainset)) if full_trainset.targets[i] in selected_classes]
test_indices = [i for i in range(len(full_testset)) if full_testset.targets[i] in selected_classes]

trainset_filtered = CustomSubset(full_trainset, train_indices, class_mapping)
testset_filtered = CustomSubset(full_testset, test_indices, class_mapping)

train_size = int(0.9 * len(trainset_filtered))
val_size = len(trainset_filtered) - train_size
trainset, valset = random_split(trainset_filtered, [train_size, val_size])

batch_size = 32
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = DataLoader(testset_filtered, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Train set size: {len(trainset)}")
print(f"Validation set size: {len(valset)}")
print(f"Test set size: {len(testset_filtered)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(1280, 5)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-06)


def save_plot(plot_func, filename):
    plot_func()
    plt.savefig(f'plots/{filename}.png')

    plt.close()


def train_model(model, trainloader, valloader, criterion, optimizer, num_epochs=20):
    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0

        train_loss = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            train_loss += loss.item()

        train_acc = correct / total
        train_acc_history.append(train_acc)

        train_loss /= len(trainloader)
        train_loss_history.append(train_loss)

        model.eval()
        correct, total = 0, 0
        val_loss = 0

        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_acc = correct / total
        val_acc_history.append(val_acc)

        val_loss /= len(valloader)
        val_loss_history.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return train_acc_history, val_acc_history, train_loss_history, val_loss_history


def evaluate_model(model, testloader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

    return test_acc


def compute_loss(model, dataloader, criterion):
    model.eval()

    total_loss, total_samples = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    return total_loss / total_samples


train_acc, val_acc, train_loss_history, val_loss_history = train_model(model, trainloader, valloader, criterion, optimizer, num_epochs=20)

save_plot(lambda: plt.plot(train_acc, label="Train") or plt.plot(val_acc, label="Validation", linestyle="dashed")
                or plt.xlabel("Epoch") or plt.ylabel("Accuracy") or plt.legend()
                or plt.title("Train/Val Accuracy"), "train_val_accuracy")

test_acc = evaluate_model(model, testloader)

train_loss = compute_loss(model, trainloader, criterion)
val_loss = compute_loss(model, valloader, criterion)
test_loss = compute_loss(model, testloader, criterion)

print(f"Train Loss: {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

save_plot(lambda: plt.bar(["Train", "Validation", "Test"], [train_loss, val_loss, test_loss], color=["blue", "orange", "red"])
                or plt.xlabel("Dataset") or plt.ylabel("Loss") or plt.title("Train/Validation/Test Loss"), "train_val_test_loss")

epochs = range(1, len(train_loss_history) + 1)
save_plot(lambda: plt.plot(epochs, train_loss_history, label="Train", color='blue')
                or plt.plot(epochs, val_loss_history, label="Validation", color='red', linestyle="dashed")
                or plt.xlabel("Epochs") or plt.ylabel("Loss") or plt.title("Train/Val Loss")
                or plt.legend() or plt.grid(), "train_val_loss")


def test_metrics(model, testloader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

    return test_acc, test_f1


test_metrics(model, testloader)
