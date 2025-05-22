import torch
import time
import copy
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from data_pre import ImageDataset, data_preprocessing


def setup_optimizer(model, lr_base=1e-4, lr_fc=1e-3):
    pretrained_prams = []
    for name, param in model.named_parameters():
        if 'fc' not in name:
            pretrained_prams.append(param)

    fc_params = list(model.fc.parameters())

    params_optimize = [
        {'params': pretrained_prams, 'lr': lr_base},
        {'params': fc_params, 'lr': lr_fc}
    ]

    optimizer = optim.Adam(params_optimize)
    return optimizer


def create_model(num_classes=101, pretrained=False):
    if pretrained:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet34(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_resnet(model, train_loader, val_loader, criterion, optimizer, device, epochs=20, log_dir='./logs', model_name='resnet34'):
    model = model.to(device)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f"{log_dir}/{model_name}_{timestamp}")

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/epochs:')

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(pred == torch.max(labels, 1)[1])

            if batch_idx % 5 == 0:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Batch/Train Loss', loss.item(), step)
                writer.add_scalar('Batch/Train Acc', torch.sum(pred == torch.max(labels, 1)[1]).item() / inputs.size(0), step)

                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Batch/Learning Rate', lr, step)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        writer.add_scalar('Epoch/Train Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/Train Acc', epoch_acc, epoch)

        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred == torch.max(labels, 1)[1])

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        scheduler.step(epoch_acc)

        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.item())

        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        writer.add_scalar('Epoch/Val Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/Val Acc', epoch_acc, epoch)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': epoch_acc,
                'history': history
            }, f"{log_dir}/{model_name}_{epoch + 1}epoch_checkpoint.pth")

    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)

    writer.close()

    return model, history

def main():
    train_transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    )

    val_transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    )

    x_train, y_train, x_test, y_test, x_val, y_val = data_preprocessing()
    train_data = ImageDataset(x_train, y_train, train_transform)
    val_data = ImageDataset(x_val, y_val, val_transform)
    test_data = ImageDataset(x_test, y_test, val_transform)

    batch_size = 16
    epochs = 100
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = create_model(num_classes=101, pretrained=True)
    optimizer = setup_optimizer(model=model, lr_base=1e-3, lr_fc=1e-3)
    criterion = nn.CrossEntropyLoss()
    model, history = train_resnet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        log_dir='./logs',
        model_name='resnet34_finetune'
    )


if __name__ == "__main__":
    main()
