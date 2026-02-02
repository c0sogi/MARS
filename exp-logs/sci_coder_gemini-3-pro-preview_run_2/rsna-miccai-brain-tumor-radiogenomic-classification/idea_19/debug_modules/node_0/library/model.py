import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from sklearn.metrics import roc_auc_score
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders


class AsymmetricEfficientNet(nn.Module):
    """
    EfficientNet-B0 with Asymmetric Grouped Convolutional Stem and Regularized Head.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # Load backbone with num_classes=1 to get the correct head dimension initially
        self.model = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=1
        )

        # --- 1. Modify Stem for 12 Channels & Grouped Convolutions ---
        old_stem = self.model.conv_stem

        # Original Stem: in=3, out=32, kernel=3, stride=2, padding=1
        # New Stem: in=12, out=32, groups=4 (Modality Isolation)
        # Weight shape for Groups=4: (Out, In/Groups, K, K) = (32, 12/4, 3, 3) = (32, 3, 3, 3)
        # This matches the original stem's weight shape exactly.

        new_stem = nn.Conv2d(
            in_channels=12,
            out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=old_stem.bias is not None,
            groups=4,
        )

        # --- 2. Asymmetric Filter Initialization ---
        if pretrained:
            # We directly copy the weights. This distributes the 32 ImageNet filters
            # across the 4 modalities (8 filters per modality).
            new_stem.weight.data = old_stem.weight.data.clone()
            if old_stem.bias is not None:
                new_stem.bias.data = old_stem.bias.data.clone()

        self.model.conv_stem = new_stem

        # --- 3. Regularized Head ---
        # Replace the default classifier with Dropout + Linear
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.model(x)


def train_model(
    epochs=10, batch_size=32, debug_limit=None, save_path="./working/best_model.pth"
):
    """
    Trains the AsymmetricEfficientNet model.
    """
    device = get_device()
    seed_everything(42)

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, debug_limit=debug_limit
    )

    # Initialize Model
    model = AsymmetricEfficientNet(pretrained=True).to(device)

    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    best_auc = 0.0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                probs = torch.sigmoid(outputs)

                # Handle potential scalar outputs if batch size is 1
                if probs.ndim == 0:
                    probs = probs.unsqueeze(0)

                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        # Compute Metric
        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            val_auc = 0.5

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)

    print(f"Training complete. Best Validation AUC: {best_auc:.6f}")
    return best_auc


def predict_and_submit(
    model_path="./working/best_model.pth", output_path="./submission/submission.csv"
):
    """
    Generates predictions for the test set using TTA and saves to CSV.
    """
    device = get_device()
    seed_everything(42)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get Test Loader (with caching enabled)
    _, _, test_loader = get_dataloaders(batch_size=32, load_cached_data=True)

    # Load Model
    model = AsymmetricEfficientNet(pretrained=False).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(
            f"Warning: Model path {model_path} not found. Using random initialization (for debugging only)."
        )

    model.eval()
    results = []

    print("Generating predictions with TTA...")

    with torch.no_grad():
        for images, subject_ids in test_loader:
            images = images.to(device)

            # --- Test-Time Augmentation (TTA) ---

            # 1. Original
            out_orig = torch.sigmoid(model(images).squeeze())

            # 2. Horizontal Flip
            images_h = torch.flip(images, [3])  # [B, C, H, W], flip W
            out_h = torch.sigmoid(model(images_h).squeeze())

            # 3. Vertical Flip
            images_v = torch.flip(images, [2])  # [B, C, H, W], flip H
            out_v = torch.sigmoid(model(images_v).squeeze())

            # Average Predictions
            avg_preds = (out_orig + out_h + out_v) / 3.0

            # Handle scalar output for batch size 1
            if avg_preds.ndim == 0:
                avg_preds = avg_preds.unsqueeze(0)

            # Store results
            for i, pid in enumerate(subject_ids):
                results.append(
                    {"BraTS21ID": pid.item(), "MGMT_value": avg_preds[i].item()}
                )

    # Save Submission
    df = pd.DataFrame(results)

    # Ensure format matches sample submission
    # Sample: BraTS21ID, MGMT_value
    df = df[["BraTS21ID", "MGMT_value"]]
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
