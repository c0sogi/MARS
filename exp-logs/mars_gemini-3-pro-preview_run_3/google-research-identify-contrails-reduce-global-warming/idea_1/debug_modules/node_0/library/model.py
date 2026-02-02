import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.data import get_dataloaders
from library.utils import rle_encode, dice_coefficient


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block:
    Two 3x3 Convolutions, each followed by Batch Normalization and ReLU.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class LightUNet(nn.Module):
    """
    Lightweight U-Net architecture for Contrail Detection.
    Optimized for speed with a compact encoder-decoder structure.
    """

    def __init__(self):
        super().__init__()
        filters = Config.ENCODER_FILTERS  # e.g., [16, 32, 64, 128]
        in_channels = Config.IN_CHANNELS

        # --- Encoder ---
        # Level 1
        self.enc1 = ConvBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)

        # Level 2
        self.enc2 = ConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)

        # Level 3
        self.enc3 = ConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(filters[2], filters[3])

        # --- Decoder ---
        # Level 3 Up
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Input: Bottleneck (128) + Enc3 (64) = 192 -> Output: 64
        self.dec1 = ConvBlock(filters[3] + filters[2], filters[2])

        # Level 2 Up
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Input: Dec1 (64) + Enc2 (32) = 96 -> Output: 32
        self.dec2 = ConvBlock(filters[2] + filters[1], filters[1])

        # Level 1 Up
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Input: Dec2 (32) + Enc1 (16) = 48 -> Output: 16
        self.dec3 = ConvBlock(filters[1] + filters[0], filters[0])

        # --- Output Head ---
        self.final_conv = nn.Conv2d(filters[0], 1, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder Path
        d1 = self.up1(b)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)

        # Output
        out = torch.sigmoid(self.final_conv(d3))
        return out


def dice_loss(preds, targets, smooth=1e-6):
    """
    Computes the differentiable Soft Dice Loss.
    """
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice


def train_model():
    """
    Trains the LightUNet model using the training set and validates on the validation set.
    Implements Early Stopping and saves the best model checkpoint.
    """
    device = torch.device(Config.DEVICE)

    # Initialize DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )

    # Initialize Model, Optimizer, Loss
    model = LightUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, verbose=False
    )
    bce_criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Combined Loss: Dice + BCE
            loss = dice_loss(outputs, masks) + bce_criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                # Loss Calculation
                loss = dice_loss(outputs, masks) + bce_criterion(outputs, masks)
                val_loss += loss.item()

                # Metric Calculation (Binary Dice)
                preds_binary = (outputs > Config.THRESHOLD).float()
                val_dice += dice_coefficient(preds_binary, masks)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss} - Val Dice: {avg_val_dice}"
        )

        # Learning Rate Scheduling
        scheduler.step(avg_val_loss)

        # --- Early Stopping & Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    return best_model_path


def inference(model_path):
    """
    Loads the trained model and generates predictions for the test set.
    Saves the result to the submission file in RLE format.
    """
    device = torch.device(Config.DEVICE)

    # Get Test DataLoader
    _, _, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )

    # Load Model
    model = LightUNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: Model weights not found. Using random initialization.")

    model.eval()

    submission_data = []

    print("Starting inference...")

    with torch.no_grad():
        for images, _, record_ids in test_loader:
            images = images.to(device)
            outputs = model(images)

            # Apply threshold to get binary mask
            preds = (outputs > Config.THRESHOLD).float().cpu().numpy()

            # Encode each image in the batch
            for i, record_id in enumerate(record_ids):
                # Extract single mask: (1, H, W) -> (H, W)
                mask = preds[i, 0, :, :]
                rle = rle_encode(mask)
                submission_data.append({"record_id": record_id, "encoded_pixels": rle})

    # Save Submission
    df = pd.DataFrame(submission_data)
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    """
    Orchestrates the full pipeline: Training followed by Inference.
    """
    best_model_path = train_model()
    inference(best_model_path)
