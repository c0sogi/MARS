import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.dataset import CameraTrapDataset


class EfficientNetClassifier(nn.Module):
    """
    EfficientNetV2-S classifier.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, fine_tune=True):
        super(EfficientNetClassifier, self).__init__()

        # Load pre-trained EfficientNetV2-S
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        self.backbone = models.efficientnet_v2_s(weights=weights)

        # Freeze backbone if not fine-tuning
        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final classification layer
        # EfficientNetV2 classifier is a Sequential, last layer is Linear
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_model(sample_size=None):
    """
    Trains the ResNetClassifier.

    Args:
        sample_size (int, optional): Number of samples to use for debugging.

    Returns:
        model: The trained PyTorch model with the best validation accuracy.
    """
    Config.set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Prepare Datasets and Loaders
    train_dataset = CameraTrapDataset(split="train", sample_size=sample_size)
    val_dataset = CameraTrapDataset(split="val", sample_size=sample_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model, Loss, Optimizer
    model = EfficientNetClassifier(num_classes=Config.NUM_CLASSES, fine_tune=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients (the fc layer)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LEARNING_RATE
    )

    best_val_acc = -1.0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    Config.make_dirs()

    print(f"Starting training on device: {device}")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        # Print metrics with full precision
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {epoch_loss}")
        print(f"Train Acc: {epoch_acc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Acc: {val_acc}")

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    # Load best model
    print(f"Loading best model with Val Acc: {best_val_acc}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def predict_and_submit(model, sample_size=None):
    """
    Generates predictions for the test set and saves to submission.csv.

    Args:
        model: Trained PyTorch model.
        sample_size (int, optional): Number of samples to use for debugging.
    """
    Config.set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    model.eval()

    test_dataset = CameraTrapDataset(split="test", sample_size=sample_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    ids = []
    predictions = []

    print("Starting inference on test set...")

    with torch.no_grad():
        for images, img_ids in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted_indices = torch.max(outputs, 1)

            ids.extend(img_ids)
            predictions.extend(predicted_indices.cpu().numpy())

    # Create DataFrame
    df_submission = pd.DataFrame({"Id": ids, "Category": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
