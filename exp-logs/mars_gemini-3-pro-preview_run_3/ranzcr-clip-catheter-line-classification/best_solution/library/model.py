import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.data import get_dataloaders
from library.utils import seed_everything, get_score


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential shape mismatch due to padding/odd dimensions
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=True
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )
        # Get channel counts from the backbone
        # EfficientNetV2-S features_only=True returns 5 feature maps
        self.feature_channels = self.encoder.feature_info.channels()
        c = self.feature_channels

        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c[-1], num_classes)

        # Segmentation Head (U-Net Decoder)
        # C4(32x) -> C3(16x) -> C2(8x) -> C1(4x) -> C0(2x) -> Out(1x)
        self.dec4 = DecoderBlock(c[4], c[3], 256)
        self.dec3 = DecoderBlock(256, c[2], 128)
        self.dec2 = DecoderBlock(128, c[1], 64)
        self.dec1 = DecoderBlock(64, c[0], 32)

        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        features = self.encoder(x)  # [c0, c1, c2, c3, c4]

        # Classification
        x_cls = self.global_pool(features[-1]).flatten(1)
        cls_logits = self.head(x_cls)

        # Segmentation
        x_seg = self.dec4(features[4], features[3])
        x_seg = self.dec3(x_seg, features[2])
        x_seg = self.dec2(x_seg, features[1])
        x_seg = self.dec1(x_seg, features[0])
        x_seg = self.final_up(x_seg)
        seg_logits = self.final_conv(x_seg)

        # Ensure output size matches input size
        if seg_logits.shape[-2:] != x.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits, size=x.shape[-2:], mode="bilinear", align_corners=True
            )

        return cls_logits, seg_logits


def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    cls_loss_sum = 0
    seg_loss_sum = 0

    cls_criterion = nn.BCEWithLogitsLoss()
    seg_criterion = nn.BCEWithLogitsLoss(reduction="none")

    for batch_idx, (images, targets, masks) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with autocast():
            cls_logits, seg_logits = model(images)

            # Classification Loss
            loss_cls = cls_criterion(cls_logits, targets)

            # Segmentation Loss (Masked)
            # We only want to supervise segmentation if we have a mask (sum > 0)
            # OR if the image is a true negative (all targets are 0)
            loss_seg_raw = seg_criterion(seg_logits, masks)  # B, 1, H, W
            loss_seg_sample = loss_seg_raw.mean(dim=(1, 2, 3))  # B

            has_mask = masks.view(masks.size(0), -1).sum(dim=1) > 0
            is_negative = targets.sum(dim=1) == 0

            # Weight is 1 if we have annotation or if it's a negative sample, else 0
            weights = (has_mask | is_negative).float()

            loss_seg = (loss_seg_sample * weights).sum() / (weights.sum() + 1e-6)

            # Total Loss
            loss = loss_cls + Config.AUX_LOSS_WEIGHT * loss_seg

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        cls_loss_sum += loss_cls.item()
        seg_loss_sum += loss_seg.item()

    avg_loss = total_loss / len(loader)
    print(
        f"Epoch {epoch+1} Train Loss: {avg_loss:.6f} (Cls: {cls_loss_sum/len(loader):.6f}, Seg: {seg_loss_sum/len(loader):.6f})"
    )
    return avg_loss


def validate(model, loader, device):
    model.eval()
    preds = []
    targets_list = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)

            cls_logits, _ = model(images)
            probs = torch.sigmoid(cls_logits)

            preds.append(probs.cpu().numpy())
            targets_list.append(targets.numpy())

    preds = np.concatenate(preds, axis=0)
    targets_list = np.concatenate(targets_list, axis=0)

    score = get_score(targets_list, preds)
    return score


def predict_test(model, loader, device):
    model.eval()
    preds = []
    uids = []

    with torch.no_grad():
        for images, batch_uids in loader:
            images = images.to(device)
            cls_logits, _ = model(images)
            probs = torch.sigmoid(cls_logits)

            preds.append(probs.cpu().numpy())
            uids.extend(batch_uids)

    preds = np.concatenate(preds, axis=0)
    return uids, preds


def main():
    seed_everything(Config.SEED)

    # Data
    train_loader, val_loader, test_loader = get_dataloaders()

    # Model
    model = MultiTaskModel()
    model.to(Config.DEVICE)

    # Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )
    scaler = GradScaler()

    best_score = 0.0
    best_epoch = 0

    # Training Loop
    print(f"Starting training on {Config.DEVICE}...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, Config.DEVICE, epoch
        )
        val_score = validate(model, val_loader, Config.DEVICE)

        print(f"Epoch {epoch+1} Val AUC: {val_score}")

        scheduler.step()

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved!")

    print(f"Training complete. Best AUC: {best_score} at Epoch {best_epoch}")

    # Inference
    print("Starting inference...")
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        )
    else:
        print("Warning: Best model not found, using current weights.")

    uids, preds = predict_test(model, test_loader, Config.DEVICE)

    # Create Submission
    df_sub = pd.DataFrame(preds, columns=Config.LABELS)
    df_sub.insert(0, "StudyInstanceUID", uids)

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
