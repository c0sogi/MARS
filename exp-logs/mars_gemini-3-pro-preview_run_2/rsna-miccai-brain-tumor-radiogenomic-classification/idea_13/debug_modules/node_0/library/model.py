import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.utils import get_device, seed_everything
from library.data_loader import get_dataloaders


class AsymmetricEfficientNet(nn.Module):
    """
    EfficientNet-B0 with Asymmetric Grouped Convolutional Stem.

    Input: (B, 12, 224, 224) - 4 modalities x 3 slices
    Stem: Grouped Conv (groups=4) to isolate modalities.
    Init: Distributes ImageNet weights across groups.
    Head: Dropout + Linear.
    """

    def __init__(self, pretrained=True, dropout_rate=0.3):
        super().__init__()
        # Load standard EfficientNet-B0
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=1
        )

        # --- Modify Stem ---
        # Original: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # New: Conv2d(12, 32, kernel_size=3, stride=2, padding=1, groups=4, bias=False)

        old_stem = self.backbone.conv_stem
        new_stem = nn.Conv2d(
            in_channels=12,
            out_channels=32,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=False,
            groups=4,
        )

        # Asymmetric Initialization:
        # The weight shape for both old and new stem is (32, 3, 3, 3).
        # Old: (Out=32, In=3, K=3, K=3)
        # New (Grouped): (Out=32, In=12//4=3, K=3, K=3)
        # We copy weights directly. This assigns Filters 0-7 to Group 1 (FLAIR),
        # Filters 8-15 to Group 2 (T1w), etc., preserving filter diversity.
        if pretrained:
            new_stem.weight.data = old_stem.weight.data.clone()

        self.backbone.conv_stem = new_stem

        # --- Modify Head ---
        # timm's efficientnet_b0.classifier is a Linear layer.
        # We wrap it with Dropout for explicit regularization.
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        # Returns logits
        return self.backbone(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Collect for metrics
        all_targets.extend(targets.detach().cpu().numpy())
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)

            all_targets.extend(targets.detach().cpu().numpy())
            all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training(
    epochs=15,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-2,
    patience=5,
    save_dir="./working/idea_13",
    debug=False,
):
    """
    Executes the training pipeline with Early Stopping.
    """
    seed_everything(42)
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, load_cached_data=True
    )

    # Initialize Model
    model = AsymmetricEfficientNet(pretrained=True)
    model.to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(save_dir, "best_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Train AUC: {train_auc} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs of no improvement."
            )
            break

    return best_auc


def predict_and_submit(
    model_path="./working/idea_13/best_model.pth",
    output_file="./submission/submission.csv",
    batch_size=32,
):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    TTA: Average of Original, Horizontal Flip, and Vertical Flip.
    """
    seed_everything(42)
    device = get_device()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Get Test Loader
    _, _, test_loader = get_dataloaders(batch_size=batch_size, load_cached_data=True)

    # Load Model
    model = AsymmetricEfficientNet(pretrained=False)  # Weights loaded from file
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(
            f"Warning: Model path {model_path} not found. Using random weights (for debugging only)."
        )

    model.to(device)
    model.eval()

    predictions = []
    ids = []

    print("Generating predictions with TTA...")

    with torch.no_grad():
        for inputs, sids in test_loader:
            inputs = inputs.to(device)

            # TTA 1: Original
            logits_orig = model(inputs)
            probs_orig = torch.sigmoid(logits_orig)

            # TTA 2: Horizontal Flip (dim 3)
            inputs_h = torch.flip(inputs, dims=[3])
            logits_h = model(inputs_h)
            probs_h = torch.sigmoid(logits_h)

            # TTA 3: Vertical Flip (dim 2)
            inputs_v = torch.flip(inputs, dims=[2])
            logits_v = model(inputs_v)
            probs_v = torch.sigmoid(logits_v)

            # Average Probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            predictions.extend(avg_probs.cpu().numpy().flatten())
            ids.extend(sids.numpy())

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Format BraTS21ID as 5-digit string for consistency if needed,
    # but sample submission uses int/string. The sample shows 00001, etc.
    # The provided sample submission reader shows int64.
    # However, the task description shows "00001".
    # We will format the ID to match the sample submission format usually required.
    # Based on sample_submission.csv provided in description:
    # BraTS21ID,MGMT_value
    # 00001,0.5
    # So we should format as string with padding.

    df_sub["BraTS21ID"] = df_sub["BraTS21ID"].apply(lambda x: f"{x:05d}")
    df_sub.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
