import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import transforms
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import os
import argparse

from data_pre import ImageDataset, data_preprocessing


def load_model(model_path, num_classes=101):
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def test_model(model, test_loader, device):
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            _, true_labels = torch.max(labels, 1)
            all_labels.extend(true_labels.cpu().numpy())

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

            running_corrects += torch.sum(preds == torch.max(labels, 1)[1])

    accuracy = running_corrects.double() / len(test_loader.dataset)

    return all_preds, all_labels, accuracy.item()


def generate_classification_report(y_true, y_pred, class_names=None, save_path=None):
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print(report)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1-score': f1
    }

    print("\nTotal metric:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    if save_path:
        with open(save_path, 'w') as f:
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nTotal metric:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")

    return metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    )

    _, _, x_test, y_test, _, _ = data_preprocessing()
    test_data = ImageDataset(x_test, y_test, test_transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.model_path, num_classes=args.num_classes)
    model = model.to(device)

    predictions, labels, accuracy = test_model(model, test_loader, device)
    print(f"Accuracy on test dataset: {accuracy:.4f}")

    os.makedirs(args.results_dir, exist_ok=True)

    class_names = [f"Class{i}" for i in range(args.num_classes)]
    report_path = os.path.join(args.results_dir, "classification_report.txt")
    metrics = generate_classification_report(labels, predictions, class_names, report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试ResNet模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--num_classes', type=int, default=101, help='分类类别数')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    parser.add_argument('--results_dir', type=str, default='./test_results', help='结果保存目录')
    args = parser.parse_args()

    main(args)
