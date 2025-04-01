import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import mobilenet_v3_large
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Subset, Dataset


BATCH_SIZE = 32


class MobileNetV3Modified(torch.nn.Module):
    def __init__(self):
        super(MobileNetV3Modified, self).__init__()
        self.model = mobilenet_v3_large(weights="IMAGENET1K_V1")
        self.model.classifier[3] = torch.nn.Linear(1280, 5)

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


def evaluate_model():
    device = torch.device("cpu")
    model = MobileNetV3Modified().to(device)

    model.load_state_dict(torch.load("mobilenetv3_cifar100.pth"))
    model.eval()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar100_metaclasses = {
        "aquatic_mammals": [4, 30, 55, 72, 95],
        "flowers": [54, 62, 70, 82, 92],
        "fruit_and_vegetables": [0, 51, 53, 57, 83],
        "household_furniture": [5, 20, 25, 84, 94],
        "vehicles_1": [8, 13, 48, 58, 90],
    }

    selected_metaclasses = list(cifar100_metaclasses.keys())
    selected_classes = sum([cifar100_metaclasses[mc] for mc in selected_metaclasses], [])

    class_mapping = {}
    for meta_idx, (meta_name, class_list) in enumerate(cifar100_metaclasses.items()):
        for class_id in class_list:
            class_mapping[class_id] = meta_idx

    full_testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    test_indices = [i for i in range(len(full_testset)) if full_testset.targets[i] in selected_classes]

    testset_filtered = CustomSubset(full_testset, test_indices, class_mapping)

    testloader = DataLoader(testset_filtered, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Test set size: {len(testset_filtered)}")

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

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix - MobileNetV3 on CIFAR-100")
    plt.savefig("confusion_matrix.png")

    print("Confusion matrix saved: confusion_matrix.png")


if __name__ == "__main__":
    evaluate_model()
