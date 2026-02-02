import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm

from library.config import (
    N_CHANNELS,
    N_CLASSES,
    DEVICE,
    MODEL_SAVE_PATH,
    SUBMISSION_FILE_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
)
from library.utils import set_seed, rle_encode, dice_coef
from library.dataset import get_dataloader

# ==========================================
# Model Architecture: Symmetric U-Net++
# ==========================================


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResNet18Encoder(nn.Module):
    """
    ResNet18 Encoder with modified input layer for N channels.
    """

    def __init__(self, in_channels=N_CHANNELS):
        super().__init__()
        # Load pre-trained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify first layer: 3 channels -> N channels
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize new weights by copying original RGB weights
        # We assume the channels are groups of 3 (Ash, Diff1, ...)
        with torch.no_grad():
            self.conv1.weight[:, :3] = original_conv1.weight
            if in_channels >= 6:
                self.conv1.weight[:, 3:6] = original_conv1.weight
            if in_channels >= 9:
                self.conv1.weight[:, 6:9] = original_conv1.weight

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        # Stem
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)  # Shape: (B, 64, 128, 128) -> Scale 1/2

        x1 = self.maxpool(x0)  # Shape: (B, 64, 64, 64) -> Scale 1/4
        x1 = self.layer1(x1)  # Shape: (B, 64, 64, 64)

        x2 = self.layer2(x1)  # Shape: (B, 128, 32, 32) -> Scale 1/8
        x3 = self.layer3(x2)  # Shape: (B, 256, 16, 16) -> Scale 1/16
        x4 = self.layer4(x3)  # Shape: (B, 512, 8, 8)   -> Scale 1/32

        return [x0, x1, x2, x3, x4]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net Architecture with ResNet18 Encoder.
    Cite solution_lesson_node_00006: Simpler U-Net outperforms U-Net++ for this task.
    """

    def __init__(self, in_channels=N_CHANNELS, n_classes=N_CLASSES):
        super().__init__()
        self.encoder = ResNet18Encoder(in_channels)

        # Encoder channels:
        # x0: 64 (Scale 1/2)
        # x1: 64 (Scale 1/4)
        # x2: 128 (Scale 1/8)
        # x3: 256 (Scale 1/16)
        # x4: 512 (Scale 1/32)

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)

        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        logits = self.final(d1)

        # Upsample to full resolution (Scale 1/1)
        logits = F.interpolate(
            logits, scale_factor=2, mode="bilinear", align_corners=True
        )

        return logits


# ==========================================
# Training & Inference Utilities
# ==========================================


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


def train_model(debug=False):
    """
    Main training loop.
    """
    set_seed()

    # Loaders
    train_loader = get_dataloader("train", batch_size=BATCH_SIZE, debug=debug)
    val_loader = get_dataloader("validation", batch_size=BATCH_SIZE, debug=debug)

    # Model Setup
    model = UNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Loss: Combo of BCE and Dice
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    best_dice = 0.0
    patience = 3
    patience_counter = 0

    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False
        ):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)

            loss_bce = criterion_bce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False
            ):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(images)

                loss_bce = criterion_bce(logits, masks)
                loss_dice = criterion_dice(logits, masks)
                loss = 0.5 * loss_bce + 0.5 * loss_dice

                val_loss += loss.item()

                # Metric calculation
                preds = torch.sigmoid(logits)
                preds = (preds > 0.5).float()
                val_dice += dice_coef(preds, masks).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Dice={avg_val_dice:.6f}"
        )

        # --- Checkpoint & Early Stopping ---
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  New best model saved! Dice: {best_dice:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val Dice: {best_dice:.6f}")


def predict_and_submit(debug=False):
    """
    Inference loop and submission generation.
    """
    set_seed()

    # Load Model
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_SAVE_PATH}")

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    # Load Test Data
    test_loader = get_dataloader("test", batch_size=BATCH_SIZE, debug=debug)

    submission_data = []

    print("Starting inference...")
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(test_loader, desc="Inference")):
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy()

            # Retrieve record_ids for this batch
            # The dataset returns (image, mask), but we need record_ids.
            # We can access the dataset from the loader.
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + images.size(0)
            batch_records = test_loader.dataset.df.iloc[start_idx:end_idx][
                "record_id"
            ].values

            for j, pred_mask in enumerate(preds):
                # pred_mask is (1, H, W), flatten to (H, W)
                pred_mask = pred_mask.squeeze(0)
                rle = rle_encode(pred_mask)
                record_id = str(batch_records[j])
                submission_data.append({"record_id": record_id, "encoded_pixels": rle})

    # Create Submission DataFrame
    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv(SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_FILE_PATH}")
