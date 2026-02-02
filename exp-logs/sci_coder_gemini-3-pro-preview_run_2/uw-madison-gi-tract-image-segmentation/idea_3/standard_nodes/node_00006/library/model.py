import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import cv2

# Import from provided library
from library.config import Config
from library.dataset import UWMadisonDataset
from library.utils import set_seed, rle_encode

# ====================================================
# Model Architecture: 2.5D DeepLabV3+ with MobileNetV2
# ====================================================


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        rate1, rate2, rate3 = atrous_rates
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(DeepLabV3Plus, self).__init__()

        # Load MobileNetV2 Backbone
        # Use weights='DEFAULT' if available, else fallback to pretrained=True
        try:
            backbone = models.mobilenet_v2(weights="DEFAULT")
        except:
            backbone = models.mobilenet_v2(pretrained=True)

        self.features = backbone.features

        # Modify Backbone for Output Stride = 16
        # MobileNetV2 normally downsamples to /32. We modify the last stride-2 block (index 14)
        # and increase dilation for subsequent blocks to maintain receptive field.

        # Layer 14 is InvertedResidual(96->160, stride=2). Change stride to 1.
        # The depthwise conv is at index 1 of the block's sequential module (expand ratio=6).
        self.features[14].conv[1][0].stride = (1, 1)

        # For layers 14 to 17, increase dilation to 2
        for i in range(14, 18):
            block = self.features[i]
            # Depthwise conv is at index 1 (expand ratio=6 for these layers)
            dw_conv = block.conv[1][0]
            dw_conv.dilation = (2, 2)
            dw_conv.padding = (2, 2)

        # Low level features (24 channels) come from layer 3 (stride 4)
        self.low_level_idx = 3
        # High level features come from the end of features (stride 16 after modification)

        # ASPP
        # MobileNetV2 last channel count is 1280
        self.aspp = ASPP(in_channels=1280, atrous_rates=[6, 12, 18])

        # Decoder
        self.low_level_project = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        # Encoder
        low_level_feat = None

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.low_level_idx:
                low_level_feat = x

        # ASPP
        x = self.aspp(x)

        # Decoder
        x = F.interpolate(
            x, size=low_level_feat.shape[-2:], mode="bilinear", align_corners=False
        )
        low_level_feat = self.low_level_project(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)
        x = F.interpolate(
            x,
            size=(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
            mode="bilinear",
            align_corners=False,
        )

        return x


# ====================================================
# Training Logic
# ====================================================


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice Coefficient.
    y_true, y_pred: (B, C, H, W)
    """
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    return (2.0 * intersection + smooth) / (
        torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + smooth
    )


def loss_fn(y_pred, y_true):
    """
    Combined BCE + Dice Loss.
    """
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true)
    pred_prob = torch.sigmoid(y_pred)
    dice = 1.0 - dice_coef(y_true, pred_prob).mean()
    return 0.5 * bce + 0.5 * dice


def train_model(load_cached_data=True):
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 1. Prepare Data
    train_dataset = UWMadisonDataset(split="train", load_cached_data=load_cached_data)
    val_dataset = UWMadisonDataset(split="val", load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model & Optimization
    model = DeepLabV3Plus(num_classes=Config.NUM_CLASSES).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    best_dice = 0.0
    patience = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # Train Loop
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)

                outputs = model(imgs)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

                preds = torch.sigmoid(outputs)
                preds = (preds > Config.CONFIDENCE_THRESHOLD).float()
                val_dice += dice_coef(masks, preds).mean().item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Dice: {val_dice:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            print(f"  Saved Best Model! Dice: {best_dice:.6f}")
            patience = 0
        else:
            patience += 1

        if patience >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break


# ====================================================
# Inference Logic
# ====================================================


def inference(load_cached_data=True):
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 1. Load Data
    test_dataset = UWMadisonDataset(split="test", load_cached_data=load_cached_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Model
    model = DeepLabV3Plus(num_classes=Config.NUM_CLASSES).to(device)
    if os.path.exists(Config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
    else:
        print("Warning: No checkpoint found. Using random weights.")

    model.eval()

    submission_rows = []
    classes = ["large_bowel", "small_bowel", "stomach"]

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            ids = batch["id"]
            orig_hs = batch["img_height"].numpy()
            orig_ws = batch["img_width"].numpy()

            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()

            for i in range(len(ids)):
                # Resize prediction back to original image size
                pred_vol = probs[i]  # (3, 256, 256)
                h, w = orig_hs[i], orig_ws[i]

                # Transpose to (H, W, C) for cv2
                pred_vol = np.transpose(pred_vol, (1, 2, 0))
                pred_vol = cv2.resize(pred_vol, (w, h), interpolation=cv2.INTER_LINEAR)

                # Threshold
                mask_vol = (pred_vol > Config.CONFIDENCE_THRESHOLD).astype(np.uint8)

                # Process each class
                for c_idx, cls_name in enumerate(classes):
                    mask = mask_vol[:, :, c_idx]

                    # 2D Cleanup (Proxy for 3D cleanup due to library constraints)
                    # Remove small connected components
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                        mask, connectivity=8
                    )
                    cleaned_mask = np.zeros_like(mask)
                    for label_idx in range(1, num_labels):
                        if (
                            stats[label_idx, cv2.CC_STAT_AREA] >= 10
                        ):  # Min area threshold
                            cleaned_mask[labels == label_idx] = 1

                    rle = rle_encode(cleaned_mask)
                    submission_rows.append(
                        {"id": ids[i], "class": cls_name, "predicted": rle}
                    )

    # 3. Save Submission
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
