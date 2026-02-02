import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config, seed_everything
from library.utils import AverageMeter, calculate_auc, save_checkpoint


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with two 3x3 convolutions and a skip connection.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # First convolution with stride
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

        # Second convolution with stride 1
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CactusResNet(nn.Module):
    """
    Custom Lightweight ResNet for 32x32 Cactus Images.
    Structure:
    - Initial Conv (3 -> 32)
    - Stage 1: 32 channels, 32x32 resolution
    - Stage 2: 64 channels, 16x16 resolution
    - Stage 3: 128 channels, 8x8 resolution
    - Global Average Pooling
    - FC Layer
    """

    def __init__(self):
        super(CactusResNet, self).__init__()
        self.in_channels = 32

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Residual Stages
        # Stage 1: 32 channels, 32x32
        self.layer1 = self._make_layer(32, blocks=2, stride=1)
        # Stage 2: 64 channels, 16x16
        self.layer2 = self._make_layer(64, blocks=2, stride=2)
        # Stage 3: 128 channels, 8x8
        self.layer3 = self._make_layer(128, blocks=2, stride=2)

        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        # First block handles stride and channel expansion
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        # Subsequent blocks are identity mappings in terms of dimensions
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=Config.NUM_EPOCHS,
    patience=Config.EARLY_STOPPING_PATIENCE,
):
    """
    Trains the model with Early Stopping and saves the best checkpoint.
    """
    seed_everything(Config.SEED)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = AverageMeter()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), images.size(0))

        # --- Validation Phase ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)

                outputs = model(images).squeeze(1)
                probs = torch.sigmoid(outputs)

                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(labels.numpy())

        val_auc = calculate_auc(val_targets, val_preds)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss.avg} - Val AUC: {val_auc}"
        )

        # --- Early Stopping & Checkpointing ---
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                Config.OUTPUT_MODEL_PATH,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

    print(f"Training complete. Best Validation AUC: {best_auc}")


def predict_and_submit(model, test_loader, device):
    """
    Generates predictions using Test Time Augmentation (TTA) and saves to submission.csv.
    TTA: Average of Original, Horizontal Flip, and Vertical Flip.
    """
    model = model.to(device)
    model.eval()

    preds = []

    # Load test metadata to get IDs
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    test_ids = test_meta["id"].values

    print("Starting inference with Test Time Augmentation...")

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # 1. Original Image
            out_orig = model(images).squeeze(1)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, [3])
            out_h = model(images_h).squeeze(1)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, [2])
            out_v = model(images_v).squeeze(1)
            prob_v = torch.sigmoid(out_v)

            # Average probabilities
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0

            preds.extend(avg_prob.cpu().numpy())

    # Create submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": preds})

    # Save to file
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
