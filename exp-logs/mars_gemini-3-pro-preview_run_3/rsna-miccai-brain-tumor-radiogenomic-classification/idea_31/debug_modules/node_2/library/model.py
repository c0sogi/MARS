import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from sklearn.metrics import roc_auc_score
from library.data_loader import get_dataloaders
from library.utils import get_device, seed_everything


class VAMGNet(nn.Module):
    """
    View-Adaptive Modality-Grouped (VAMG) Network.

    Architecture:
        - Backbone: EfficientNet-B0 (pretrained)
        - Input Channels: 64 (16 slices x 4 modalities)
        - Regularization: Drop Path Rate = 0.2
        - Head: Binary Classification (Logits)

    The input tensor shape is expected to be (B, 64, 256, 256).
    """

    def __init__(
        self,
        model_name="efficientnet_b0",
        pretrained=True,
        in_chans=64,
        num_classes=1,
        drop_path_rate=0.2,
    ):
        super(VAMGNet, self).__init__()
        # Initialize the backbone using timm
        # timm handles the adaptation of the first layer weights for in_chans != 3
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        # Forward pass through the backbone
        # x shape: (Batch_Size, 64, 256, 256)
        # Output shape: (Batch_Size, 1) - Logits
        return self.backbone(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Apply sigmoid for AUC calculation
        probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        all_preds.extend(probs)
        all_targets.extend(targets.detach().cpu().numpy().flatten())

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle case with single class in batch
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Performs validation loop.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy().flatten())

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training(
    epochs=15, batch_size=32, lr=1e-4, patience=5, save_path="./working/best_model.pth"
):
    """
    Main training loop with Early Stopping.
    """
    seed_everything(42)
    device = get_device()

    # Load Data (using cached data if available)
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, load_cached_data=True
    )

    # Initialize Model
    model = VAMGNet().to(device)

    # Optimizer & Loss
    # Using Adam with lr=1e-4 as specified
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Train AUC: {train_auc} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping and Model Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with AUC: {best_auc}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return best_auc


def generate_submission(
    model_path="./working/best_model.pth",
    output_path="./submission/submission.csv",
    batch_size=32,
):
    """
    Generates predictions for the test set and saves to CSV.
    """
    device = get_device()

    # Load Test Data
    _, _, test_loader = get_dataloaders(batch_size=batch_size, load_cached_data=True)

    # Load Model
    model = VAMGNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(
            f"Warning: Model file {model_path} not found. Predictions will be random."
        )

    model.eval()
    ids = []
    preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            batch_ids = batch["BraTS21ID"]

            outputs = model(images)
            # Apply Sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            ids.extend(batch_ids)
            preds.extend(probs)

    # Create Submission DataFrame
    df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def main():
    """
    Entry point for training and submission generation.
    """
    # Ensure working directory exists
    os.makedirs("./working", exist_ok=True)

    # Run Training
    run_training(epochs=15, batch_size=32, lr=1e-4, patience=5)

    # Generate Submission
    generate_submission()
