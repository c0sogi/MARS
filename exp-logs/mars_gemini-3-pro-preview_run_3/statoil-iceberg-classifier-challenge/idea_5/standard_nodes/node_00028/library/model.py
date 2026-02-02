import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from library.utils import get_device

# -----------------------------------------------------------------------------
# Neural Network Architecture
# -----------------------------------------------------------------------------


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Module.
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    """
    Residual Block with Squeeze-and-Excitation.
    Structure: Conv-BN-ReLU-Conv-BN-SE + Skip Connection.
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SELayer(
            out_channels, reduction=8
        )  # Reduced reduction ratio for small channels
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class MicroResNet(nn.Module):
    """
    Custom Shallow Residual Network (Micro-ResNet) for Iceberg Detection.
    Features:
    - 3-channel input (HH, HV, Avg)
    - 3 Stages of Residual Blocks (64 -> 128 -> 128 filters)
    - Global Max Pooling
    - Incidence Angle Fusion
    """

    def __init__(self):
        super(MicroResNet, self).__init__()

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stages
        # Stage 1: 64 filters
        self.layer1 = self._make_layer(64, 64, stride=1)
        # Stage 2: 128 filters (downsample)
        self.layer2 = self._make_layer(64, 128, stride=2)
        # Stage 3: 128 filters (downsample)
        self.layer3 = self._make_layer(128, 128, stride=2)

        # Classification Head
        # 128 (image features) + 1 (angle) = 129
        self.fc1 = nn.Linear(128 + 1, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        return ResBlock(in_channels, out_channels, stride, downsample)

    def forward(self, x, angle):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Backbone
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global Max Pooling
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # Flatten: [B, 128]

        # Fusion
        angle = angle.view(-1, 1)  # [B, 1]
        x = torch.cat([x, angle], dim=1)  # [B, 129]

        # Head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # Output probability
        x = torch.sigmoid(x)
        return x.squeeze(1)


# -----------------------------------------------------------------------------
# Training Logic
# -----------------------------------------------------------------------------


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=1e-3,
    patience=10,
    save_path="model.pth",
):
    """
    Trains the MicroResNet model with Early Stopping.

    Args:
        model: The MicroResNet instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Maximum number of epochs.
        lr: Learning rate (constant).
        patience: Early stopping patience.
        save_path: Path to save the best model weights.

    Returns:
        model: The trained model with best weights loaded.
        history: Dictionary containing training history.
    """
    device = get_device()
    model = model.to(device)

    # Optimizer: Adam with constant learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss: Binary Cross Entropy
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"Starting training on {device} for {epochs} epochs.")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                angles = batch["angle"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images, angles)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        history["val_loss"].append(epoch_val_loss)

        # --- Logging ---
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.8f} - Val Loss: {epoch_val_loss:.8f}"
        )

        # --- Early Stopping & Checkpointing ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    # Load best model
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"Loaded best model from {save_path} with Val Loss: {best_val_loss:.8f}")

    return model, history


# -----------------------------------------------------------------------------
# Submission Logic
# -----------------------------------------------------------------------------


def generate_submission(model, test_loader, output_path="./submission/submission.csv"):
    """
    Generates predictions using Test-Time Augmentation (TTA) and saves to CSV.

    Args:
        model: Trained MicroResNet model.
        test_loader: DataLoader for test data.
        output_path: Path to save the submission CSV.
    """
    device = get_device()
    model = model.to(device)
    model.eval()

    predictions = []
    ids = []

    print("Generating predictions with TTA...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            batch_ids = batch["id"]

            # TTA: Original
            pred_orig = model(images, angles)

            # TTA: Horizontal Flip
            images_h = torch.flip(images, [3])
            pred_h = model(images_h, angles)

            # TTA: Vertical Flip
            images_v = torch.flip(images, [2])
            pred_v = model(images_v, angles)

            # Average predictions
            pred_avg = (pred_orig + pred_h + pred_v) / 3.0

            predictions.extend(pred_avg.cpu().numpy())
            ids.extend(batch_ids)

    # Create DataFrame
    df = pd.DataFrame({"id": ids, "is_iceberg": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
