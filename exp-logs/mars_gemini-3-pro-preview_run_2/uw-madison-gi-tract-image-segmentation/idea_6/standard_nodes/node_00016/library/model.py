import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import cv2
from scipy import ndimage

from library.config import Config
from library.utils import set_seed, dice_coefficient, rle_encode
from library.dataset import process_metadata, UWDataset, get_transforms

# ---------------------------------------------------------
# 1. Model Architecture (LinkNet)
# ---------------------------------------------------------


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 Encoder for LinkNet.
    Extracts features at multiple scales.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        # Load pretrained ResNet18
        # Using weights="DEFAULT" for best available ImageNet weights
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Handle input channels
        if in_channels != 3:
            # If input is not 3 channels, replace the first conv layer
            # (Not strictly needed for this task as Config.IN_CHANNELS is 3)
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = backbone.conv1

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # 64 channels
        self.layer2 = backbone.layer2  # 128 channels
        self.layer3 = backbone.layer3  # 256 channels
        self.layer4 = backbone.layer4  # 512 channels

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        e0 = x  # Stride 2, 64ch (Used for skip in dec1)

        x = self.maxpool(x)  # Stride 4

        e1 = self.layer1(x)  # Stride 4, 64ch (Used for skip in dec2)
        e2 = self.layer2(e1)  # Stride 8, 128ch (Used for skip in dec3)
        e3 = self.layer3(e2)  # Stride 16, 256ch (Used for skip in dec4)
        e4 = self.layer4(e3)  # Stride 32, 512ch (Bottleneck)

        return e0, e1, e2, e3, e4


class DecoderBlock(nn.Module):
    """
    LinkNet Decoder Block with Additive Skip Connections.
    Structure: 1x1 Conv (reduce) -> Transposed Conv (upsample) -> 1x1 Conv (expand) -> Add Skip
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Internal channels usually reduced by factor of 4 for efficiency
        internal_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1. Reduce dimensions
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 2. Upsample (Stride 2)
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 3. Expand dimensions
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        out = self.block(x)
        if skip is not None:
            # Additive Skip Connection
            out = out + skip
        return out


class LinkNet(nn.Module):
    """
    LinkNet Architecture for Segmentation.
    Combines ResNet18Encoder and DecoderBlocks.
    """

    def __init__(self):
        super().__init__()
        self.encoder = ResNet18Encoder(in_channels=Config.IN_CHANNELS)

        # Decoder blocks
        # e4 (512) -> d4 (256) + skip e3 (256)
        self.dec4 = DecoderBlock(512, 256)
        # d4 (256) -> d3 (128) + skip e2 (128)
        self.dec3 = DecoderBlock(256, 128)
        # d3 (128) -> d2 (64) + skip e1 (64)
        self.dec2 = DecoderBlock(128, 64)
        # d2 (64) -> d1 (64) + skip e0 (64)
        self.dec1 = DecoderBlock(64, 64)

        # Final Head
        # d1 is 64ch, stride 2. We need to upsample to stride 1 and map to classes.
        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, Config.NUM_CLASSES, kernel_size=1),
        )

    def forward(self, x):
        # Encoder
        e0, e1, e2, e3, e4 = self.encoder(x)

        # Decoder
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)

        # Head
        out = self.final(d1)
        return out


# ---------------------------------------------------------
# 2. Loss Function
# ---------------------------------------------------------


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sigmoid = torch.sigmoid(pred)
        dice_loss = 1.0 - dice_coefficient(pred_sigmoid, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------
# 3. Training Logic
# ---------------------------------------------------------


def train_model():
    set_seed(Config.SEED)

    # 1. Data Preparation
    print("Processing metadata for training...")
    train_df = process_metadata(Config.TRAIN_METADATA_PATH, mode="train")
    val_df = process_metadata(Config.VAL_METADATA_PATH, mode="val")

    train_dataset = UWDataset(
        train_df, mode="train", transforms=get_transforms("train")
    )
    val_dataset = UWDataset(val_df, mode="val", transforms=get_transforms("val"))

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

    # 2. Model Setup
    model = LinkNet().to(Config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )
    criterion = BCEDiceLoss(Config.BCE_WEIGHT, Config.DICE_WEIGHT)

    best_dice = 0.0
    print(f"Starting training for {Config.EPOCHS} epochs on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0

        # Training Loop
        for images, masks in train_loader:
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # Validation Loop
        model.eval()
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(Config.DEVICE)
                masks = masks.to(Config.DEVICE)

                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > Config.PRED_THRESHOLD).float()

                # Compute Dice
                d = dice_coefficient(outputs, masks)
                val_dice += d.item() * images.size(0)

        val_dice /= len(val_dataset)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss:.6f} - Val Dice: {val_dice:.6f}"
        )

        # Save Best Model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print("  New best model saved!")

    print(f"Training complete. Best Val Dice: {best_dice:.6f}")


# ---------------------------------------------------------
# 4. Inference Logic
# ---------------------------------------------------------


def inference():
    set_seed(Config.SEED)

    # 1. Load Data
    print("Processing metadata for inference...")
    test_df = process_metadata(Config.TEST_METADATA_PATH, mode="test")
    test_dataset = UWDataset(test_df, mode="test", transforms=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Model
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Model file not found at {Config.MODEL_PATH}. Skipping inference.")
        return

    model = LinkNet().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()

    # 3. Predict & Cache
    preds_cache = []
    print("Running inference...")

    with torch.no_grad():
        for images, ids, sizes in test_loader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)

            # Move to CPU
            outputs = outputs.cpu().numpy()

            for i in range(len(ids)):
                slice_id = ids[i]
                orig_h, orig_w = sizes[i]
                pred_mask = outputs[i]  # (C, H, W)

                # Resize back to original size
                # Transpose to (H, W, C) for cv2.resize
                pred_mask = np.transpose(pred_mask, (1, 2, 0))
                # cv2.resize takes (width, height)
                pred_mask = cv2.resize(pred_mask, (int(orig_w), int(orig_h)))
                # Transpose back to (C, H, W)
                pred_mask = np.transpose(pred_mask, (2, 0, 1))

                preds_cache.append({"id": slice_id, "prediction": pred_mask})

    # 4. Post-processing (3D Connected Components)
    print("Post-processing with 3D Connected Components...")

    # Helper to parse ID for grouping
    def get_case_day(row_id):
        # id format: case123_day20_slice_0001
        parts = row_id.split("_")
        case = parts[0]
        day = parts[1]
        slice_num = int(parts[3])
        return f"{case}_{day}", slice_num

    # Group by case_day
    case_groups = {}
    for item in preds_cache:
        cd, sn = get_case_day(item["id"])
        if cd not in case_groups:
            case_groups[cd] = []
        case_groups[cd].append((sn, item))

    final_submission = []

    for case_id, items in case_groups.items():
        # Sort by slice number to form a proper volume
        items.sort(key=lambda x: x[0])

        # Get dimensions
        sample_pred = items[0][1]["prediction"]
        C, H, W = sample_pred.shape
        D = len(items)

        # Create volume (C, D, H, W)
        volume = np.zeros((C, D, H, W), dtype=np.float32)
        slice_ids = []

        for z, (sn, item) in enumerate(items):
            volume[:, z, :, :] = item["prediction"]
            slice_ids.append(item["id"])

        # Threshold
        volume_bin = (volume > Config.PRED_THRESHOLD).astype(np.uint8)

        # 3D Connected Components per class
        if Config.USE_3D_CONNECTED_COMPONENTS:
            for c in range(C):
                class_vol = volume_bin[c]
                if class_vol.sum() > 0:
                    # Label connected components
                    labeled, num_features = ndimage.label(class_vol)
                    if num_features > 1:
                        # Find largest component
                        sizes = ndimage.sum(
                            class_vol, labeled, range(1, num_features + 1)
                        )
                        largest_label = np.argmax(sizes) + 1
                        # Keep only largest
                        class_vol = (labeled == largest_label).astype(np.uint8)
                        volume_bin[c] = class_vol

        # Encode RLE for each slice
        for z in range(D):
            current_id = slice_ids[z]
            for c, class_name in enumerate(Config.CLASS_LABELS):
                mask_slice = volume_bin[c, z, :, :]
                rle = rle_encode(mask_slice)
                final_submission.append(
                    {"id": current_id, "class": class_name, "predicted": rle}
                )

    # 5. Save Submission
    sub_df = pd.DataFrame(final_submission)
    # Ensure correct column order
    sub_df = sub_df[["id", "class", "predicted"]]
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    """
    Main entry point to execute training and inference.
    """
    train_model()
    inference()
