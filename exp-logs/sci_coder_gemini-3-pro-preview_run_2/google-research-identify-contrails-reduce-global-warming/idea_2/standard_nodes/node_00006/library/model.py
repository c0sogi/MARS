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
    ResNet18 Encoder with modified input layer for 9 channels.
    """

    def __init__(self, in_channels=9):
        super().__init__()
        # Load pre-trained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify first layer: 3 channels -> 9 channels
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize new weights by averaging original RGB weights
        # We assume the 9 channels are 3 groups of 3 (Ash, Diff1, Diff2)
        with torch.no_grad():
            self.conv1.weight[:, :3] = original_conv1.weight
            self.conv1.weight[:, 3:6] = original_conv1.weight
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


class SymmetricUNetPlusPlus(nn.Module):
    """
    U-Net++ Architecture with ResNet18 Encoder.
    """

    def __init__(self, in_channels=N_CHANNELS, n_classes=N_CLASSES):
        super().__init__()
        self.encoder = ResNet18Encoder(in_channels)

        # Filter sizes for decoder levels 0 to 4
        # Level 0 corresponds to x0 (Scale 1/2)
        filters = [64, 64, 128, 256, 512]

        # --- Decoder Nodes ---
        # Row 0 (Scale 1/2)
        # Inputs: [Previous Node in Row] + [Upsampled Node from Row Below]
        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv0_2 = ConvBlock(filters[0] * 2 + filters[1], filters[0])
        self.conv0_3 = ConvBlock(filters[0] * 3 + filters[1], filters[0])
        self.conv0_4 = ConvBlock(filters[0] * 4 + filters[1], filters[0])

        # Row 1 (Scale 1/4)
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv1_2 = ConvBlock(filters[1] * 2 + filters[2], filters[1])
        self.conv1_3 = ConvBlock(filters[1] * 3 + filters[2], filters[1])

        # Row 2 (Scale 1/8)
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])
        self.conv2_2 = ConvBlock(filters[2] * 2 + filters[3], filters[2])

        # Row 3 (Scale 1/16)
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3])

        # Final Projection
        self.final_conv = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder Features
        x0_0, x1_0, x2_0, x3_0, x4_0 = self.encoder(x)

        # --- Decoder Nested Pathways ---

        # Column 1
        # x3_1: Input x3_0, Up(x4_0)
        x3_1 = self.conv3_1(
            torch.cat(
                [
                    x3_0,
                    F.interpolate(
                        x4_0, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        # x2_1: Input x2_0, Up(x3_0)
        x2_1 = self.conv2_1(
            torch.cat(
                [
                    x2_0,
                    F.interpolate(
                        x3_0, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        # x1_1: Input x1_0, Up(x2_0)
        x1_1 = self.conv1_1(
            torch.cat(
                [
                    x1_0,
                    F.interpolate(
                        x2_0, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        # x0_1: Input x0_0, Up(x1_0)
        x0_1 = self.conv0_1(
            torch.cat(
                [
                    x0_0,
                    F.interpolate(
                        x1_0, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # Column 2
        # x2_2: Input x2_0, x2_1, Up(x3_1)
        x2_2 = self.conv2_2(
            torch.cat(
                [
                    x2_0,
                    x2_1,
                    F.interpolate(
                        x3_1, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        # x1_2: Input x1_0, x1_1, Up(x2_1)
        x1_2 = self.conv1_2(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    F.interpolate(
                        x2_1, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        # x0_2: Input x0_0, x0_1, Up(x1_1)
        x0_2 = self.conv0_2(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    F.interpolate(
                        x1_1, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # Column 3
        # x1_3: Input x1_0, x1_1, x1_2, Up(x2_2)
        x1_3 = self.conv1_3(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    x1_2,
                    F.interpolate(
                        x2_2, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        # x0_3: Input x0_0, x0_1, x0_2, Up(x1_2)
        x0_3 = self.conv0_3(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    F.interpolate(
                        x1_2, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # Column 4
        # x0_4: Input x0_0, x0_1, x0_2, x0_3, Up(x1_3)
        x0_4 = self.conv0_4(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    x0_3,
                    F.interpolate(
                        x1_3, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # Final Output (Scale 1/2 -> 128x128)
        logits = self.final_conv(x0_4)

        # Upsample to full resolution (Scale 1/1 -> 256x256)
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
    model = SymmetricUNetPlusPlus().to(DEVICE)
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

    model = SymmetricUNetPlusPlus().to(DEVICE)
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
