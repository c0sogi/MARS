import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import AverageMeter, seed_everything
from library.dataset import SETIDataset


class SiameseEfficientNet(nn.Module):
    """
    Siamese Network with Explicit Difference Aggregation.
    Uses EfficientNet-B0 backbone to extract features from On and Off target streams.
    Computes a difference map and aggregates features via GAP and GMP.
    Cite solution_lesson_node_00038: Concatenate explicit difference instead of gating.
    Cite solution_lesson_node_00027: Use GMP on source streams.
    """

    def __init__(self, backbone_name=Config.BACKBONE_NAME, pretrained=True):
        super().__init__()
        # Load backbone with features_only=True to get feature maps
        # in_chans=3 because inputs are stacks of 3 spectrograms (e.g., A-A-A or B-C-D)
        # Cite solution_lesson_node_00035: Extract features from block output, not projection.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,
            features_only=True,
            out_indices=(4,),  # Use the last convolutional block (Stride 32)
        )

        # EfficientNet-B0 last conv block has 320 channels
        self.feature_dim = 320

        # We pool 3 feature maps: On, Off, Diff
        # Each uses GAP + GMP (2 stats)
        # Total input to FC = 3 * 2 * 320 = 1920
        self.fc = nn.Linear(self.feature_dim * 6, 1)

    def forward(self, on_source, off_source):
        # Shared Backbone Feature Extraction
        # Shapes: (B, 3, H, W) -> (B, 320, H/32, W/32)
        # timm features_only returns a list, we take the first (and only) element requested
        feat_on = self.backbone(on_source)[0]
        feat_off = self.backbone(off_source)[0]

        # 1. Explicit Difference (Cite solution_lesson_node_00019)
        feat_diff = feat_on - feat_off

        # 2. Hybrid Pooling Helper
        def pool_features(x):
            gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
            gmp = F.adaptive_max_pool2d(x, 1).flatten(1)
            return torch.cat([gap, gmp], dim=1)

        # Pool all representations
        p_on = pool_features(feat_on)
        p_off = pool_features(feat_off)
        p_diff = pool_features(feat_diff)

        # 3. Concatenation
        combined = torch.cat([p_on, p_off, p_diff], dim=1)

        # 4. Classification
        logits = self.fc(combined)
        return logits


def mixup_data(on, off, y, alpha=0.2, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = on.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_on = lam * on + (1 - lam) * on[index, :]
    mixed_off = lam * off + (1 - lam) * off[index, :]
    y_a, y_b = y, y[index]
    return mixed_on, mixed_off, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    losses = AverageMeter()

    for i, (on_img, off_img, target) in enumerate(loader):
        on_img = on_img.to(device)
        off_img = off_img.to(device)
        target = target.to(device).unsqueeze(1)

        # Apply Mixup
        mixed_on, mixed_off, target_a, target_b, lam = mixup_data(
            on_img, off_img, target, Config.MIXUP_ALPHA, device
        )

        optimizer.zero_grad()
        logits = model(mixed_on, mixed_off)
        loss = mixup_criterion(criterion, logits, target_a, target_b, lam)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), on_img.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for on_img, off_img, target in loader:
            on_img = on_img.to(device)
            off_img = off_img.to(device)
            target = target.to(device).unsqueeze(1)

            logits = model(on_img, off_img)
            loss = criterion(logits, target)

            losses.update(loss.item(), on_img.size(0))

            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return losses.avg, auc


def train_and_predict():
    """
    Main execution pipeline:
    1. Setup and Data Loading
    2. Training Loop with Mixup and Early Stopping
    3. Inference with TTA
    4. Submission Generation
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # --- Data Loading ---
    print("Loading Metadata...")
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    train_dataset = SETIDataset(df_train, mode="train")
    val_dataset = SETIDataset(df_val, mode="val")

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

    # --- Model Setup ---
    print("Initializing Model...")
    model = SiameseGatedEfficientNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=1e-6
    )

    criterion = nn.BCEWithLogitsLoss()

    # --- Training Loop ---
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with AUC: {val_auc}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # --- Inference with TTA ---
    print("Starting Inference with TTA...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    test_dataset = SETIDataset(df_test, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # TTA Logic
    # We perform 4 passes: Original, H-Flip, V-Flip, HV-Flip
    # Note: Dataset pads height to 288. Valid data is in [0:273].
    # When flipping vertically (freq), we must only flip the valid region to keep padding at bottom.

    h_valid = 273

    all_preds = []

    with torch.no_grad():
        for on_img, off_img, _ in test_loader:
            on_img = on_img.to(device)
            off_img = off_img.to(device)

            # 1. Original
            logit1 = model(on_img, off_img)
            pred1 = torch.sigmoid(logit1)

            # 2. Horizontal Flip (Time - Axis 3)
            # Width is 256 (full), so simple flip is safe
            logit2 = model(on_img.flip(3), off_img.flip(3))
            pred2 = torch.sigmoid(logit2)

            # 3. Vertical Flip (Frequency - Axis 2)
            # Must slice valid region, flip, then place back (or pad back)
            # Since tensor is already padded to 288, we modify in place or clone
            on_v = on_img.clone()
            off_v = off_img.clone()

            on_v[:, :, :h_valid, :] = on_v[:, :, :h_valid, :].flip(2)
            off_v[:, :, :h_valid, :] = off_v[:, :, :h_valid, :].flip(2)

            logit3 = model(on_v, off_v)
            pred3 = torch.sigmoid(logit3)

            # 4. H + V Flip
            on_hv = on_v.flip(3)  # Already V-flipped, now H-flip
            off_hv = off_v.flip(3)

            logit4 = model(on_hv, off_hv)
            pred4 = torch.sigmoid(logit4)

            # Average predictions
            avg_pred = (pred1 + pred2 + pred3 + pred4) / 4.0
            all_preds.extend(avg_pred.cpu().numpy().flatten())

    # --- Submission ---
    df_test["target"] = all_preds
    df_test[["id", "target"]].to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
