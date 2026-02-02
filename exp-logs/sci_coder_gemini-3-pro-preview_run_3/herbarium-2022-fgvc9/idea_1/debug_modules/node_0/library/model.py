import os
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import compute_metrics
from library.dataset import get_dataloaders


class PlantClassifier(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(PlantClassifier, self).__init__()
        # Load MobileNetV3 Large pretrained on ImageNet
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.model = models.mobilenet_v3_large(weights=weights)

        # Modify the classifier head
        # The classifier in MobileNetV3 is a Sequential block.
        # We need to replace the last Linear layer which maps to 1000 classes.
        # The structure is typically: Linear -> Hardswish -> Dropout -> Linear
        last_layer_idx = len(self.model.classifier) - 1
        in_features = self.model.classifier[last_layer_idx].in_features

        # Replace with new Linear layer for our number of classes
        self.model.classifier[last_layer_idx] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.scale(optimizer).step()
        scaler.update()

        if scheduler:
            scheduler.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / len(dataloader.dataset)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate Macro F1
    f1 = compute_metrics(all_labels, all_preds)

    return epoch_loss, f1


def train_model(train_loader, val_loader, epochs=Config.NUM_EPOCHS):
    device = Config.DEVICE
    print(f"Training on device: {device}")

    model = PlantClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    scaler = torch.cuda.amp.GradScaler()

    best_f1 = 0.0
    patience = 3
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler
        )
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val F1: {val_f1}"
        )

        # Early Stopping and Model Checkpointing
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Load best model for return
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    return model


def predict_and_submit(model, test_loader):
    device = Config.DEVICE
    model.eval()

    predictions = []
    image_ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            predictions.extend(preds)
            image_ids.extend(ids)

    df = pd.DataFrame({"Id": image_ids, "Predicted": predictions})

    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    # Setup directories
    Config.setup()

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # Train
    model = train_model(train_loader, val_loader)

    # Predict
    predict_and_submit(model, test_loader)
