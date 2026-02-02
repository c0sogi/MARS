import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_roc_auc, set_seed
from library.data import get_dataloaders


class SpatialDifferenceCNN(nn.Module):
    """
    A lightweight CNN designed to operate on single-channel difference maps.
    Architecture:
        - 4 Convolutional Blocks (Conv -> BN -> ReLU -> MaxPool)
        - Global Average Pooling
        - Linear Classifier
    """

    def __init__(self):
        super(SpatialDifferenceCNN, self).__init__()

        # Input channels from Config (expected 1)
        in_channels = Config.INPUT_SHAPE[0]

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        logits = self.classifier(x)
        return logits


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    # Collect targets and preds for training AUC (optional but good for monitoring)
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # Shape (batch, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

        # Store for metric calculation
        probs = torch.sigmoid(logits)
        all_targets.append(targets.detach().cpu())
        all_preds.append(probs.detach().cpu())

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()
    auc = calculate_roc_auc(all_targets, all_preds)

    return losses.avg, auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, targets)

            losses.update(loss.item(), images.size(0))

            probs = torch.sigmoid(logits)
            all_targets.append(targets.cpu())
            all_preds.append(probs.cpu())

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    auc = calculate_roc_auc(all_targets, all_preds)

    return losses.avg, auc


def train_model():
    """
    Main training routine.
    Initializes data, model, and runs the training loop with early stopping.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")

    # Data Loaders
    train_loader, val_loader, _ = get_dataloaders()

    # Model Setup
    model = SpatialDifferenceCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Training Loop Variables
    best_val_auc = 0.0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(Config.MAX_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{Config.MAX_EPOCHS}")
        print(f"  Train Loss: {train_loss:.10f} | Train AUC: {train_auc:.10f}")
        print(f"  Val Loss:   {val_loss:.10f} | Val AUC:   {val_auc:.10f}")

        # Early Stopping & Checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved to {Config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(
        f"Training complete. Best Val AUC: {best_val_auc:.10f} at epoch {best_epoch+1}"
    )


def predict_and_submit():
    """
    Loads the best model, performs inference on the test set, and saves the submission file.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Test Data
    _, _, test_loader = get_dataloaders()

    # Load Model
    model = SpatialDifferenceCNN().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Train first."
        )

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    print("Starting inference on test set...")

    ids = []
    predictions = []

    with torch.no_grad():
        for images, sample_ids in test_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            # Flatten predictions to 1D array
            probs_np = probs.cpu().numpy().flatten()

            ids.extend(sample_ids)
            predictions.extend(probs_np)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": ids, "target": predictions})

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {df_sub.shape}")
    print(f"First 5 predictions:\n{df_sub.head()}")
