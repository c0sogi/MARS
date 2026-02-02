import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

from library.config import Config
from library.utils import (
    AverageMeter,
    get_score,
    mixup_data,
    mixup_criterion,
    seed_everything,
)
from library.dataset import get_train_val_loaders, get_test_loader

# Suppress warnings
warnings.filterwarnings("ignore")


class SiameseDifferenceNet(nn.Module):
    """
    Siamese Network with EfficientNet-B0 backbone.
    Computes explicit spatial difference between On-Target and Off-Target feature maps.
    Uses Hybrid Pooling (GAP + GMP) to capture both global context and local anomalies.
    """

    def __init__(self, backbone_name=Config.BACKBONE, pretrained=Config.PRETRAINED):
        super().__init__()
        # Load backbone with no classifier and no global pooling to get spatial feature maps
        # in_chans=3 matches the stacked On/Off target input shape
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=Config.IN_CHANNELS,
        )

        # Determine feature dimension dynamically
        # EfficientNet-B0 typically has 1280 channels at the final layer
        dummy_input = torch.randn(
            1, Config.IN_CHANNELS, Config.IMG_HEIGHT, Config.IMG_WIDTH
        )
        with torch.no_grad():
            features = self.backbone(dummy_input)
        self.feature_dim = features.shape[1]

        # Pooling layers
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.global_max = nn.AdaptiveMaxPool2d(1)

        # Classifier Head
        # Concatenating 4 vectors:
        # 1. GAP(On-Target)
        # 2. GAP(Off-Target)
        # 3. GAP(Difference)
        # 4. GMP(Difference)
        self.fc = nn.Linear(self.feature_dim * 4, Config.NUM_CLASSES)

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, on_input, off_input):
        # Extract spatial features for both streams using shared backbone
        # Shape: (B, C, H', W')
        f_on = self.forward_features(on_input)
        f_off = self.forward_features(off_input)

        # Explicit Spatial Difference
        f_diff = f_on - f_off

        # Hybrid Pooling
        # Flatten(1) converts (B, C, 1, 1) -> (B, C)
        v_on = self.global_avg(f_on).flatten(1)
        v_off = self.global_avg(f_off).flatten(1)
        v_diff_avg = self.global_avg(f_diff).flatten(1)
        v_diff_max = self.global_max(f_diff).flatten(1)

        # Concatenate all context and difference vectors
        combined = torch.cat([v_on, v_off, v_diff_avg, v_diff_max], dim=1)

        # Final classification
        return self.fc(combined)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    losses = AverageMeter()

    for on_imgs, off_imgs, targets in loader:
        on_imgs = on_imgs.to(device)
        off_imgs = off_imgs.to(device)
        targets = targets.to(device)

        # Apply Mixup
        # Note: We mix both on_imgs and off_imgs with the same lambda/indices
        # to preserve the relationship between the streams.
        mixed_on, mixed_off, y_a, y_b, lam = mixup_data(
            on_imgs, off_imgs, targets, alpha=Config.MIXUP_ALPHA, device=device
        )

        optimizer.zero_grad()

        # Forward pass
        preds = model(mixed_on, mixed_off).squeeze(1)

        # Calculate Mixup Loss
        loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), on_imgs.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for on_imgs, off_imgs, targets in loader:
            on_imgs = on_imgs.to(device)
            off_imgs = off_imgs.to(device)
            targets = targets.to(device)

            # Forward pass (No Mixup)
            logits = model(on_imgs, off_imgs).squeeze(1)
            loss = criterion(logits, targets)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            losses.update(loss.item(), on_imgs.size(0))
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    auc = get_score(all_targets, all_preds)
    return losses.avg, auc


def run_training():
    """
    Main training orchestration function.
    Handles data loading, model initialization, training loop,
    early stopping, and model saving.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Using device: {device}")

    # Data Loaders
    train_loader, val_loader = get_train_val_loaders(debug=Config.DEBUG)

    # Model
    model = SiameseDifferenceNet().to(device)

    # Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = nn.BCEWithLogitsLoss()

    # Training Loop with Early Stopping
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.0f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.9f}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  -> New Best AUC! Model saved to {Config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc:.9f}")


def inference():
    """
    Generates predictions for the test set using the best saved model.
    Applies Test Time Augmentation (TTA).
    Saves results to submission.csv.
    """
    device = Config.DEVICE
    print("\nStarting inference...")

    # Load Metadata to ensure ID alignment
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_METADATA}")

    df_test = pd.read_csv(Config.TEST_METADATA)

    # Load Data
    test_loader = get_test_loader()

    # Load Model
    model = SiameseDifferenceNet().to(device)
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Best model not found at {Config.BEST_MODEL_PATH}. Run training first."
        )

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []

    with torch.no_grad():
        for on_imgs, off_imgs, _ in test_loader:
            on_imgs = on_imgs.to(device)
            off_imgs = off_imgs.to(device)

            # TTA Strategy:
            # 1. Original Input
            logits_orig = model(on_imgs, off_imgs).squeeze(1)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Flipped Input (Horizontal + Vertical)
            # Dims are (B, C, H, W). Flip H (dim 2) and W (dim 3).
            # This corresponds to Frequency Inversion + Time Reversal.
            on_imgs_flip = torch.flip(on_imgs, [2, 3])
            off_imgs_flip = torch.flip(off_imgs, [2, 3])

            logits_flip = model(on_imgs_flip, off_imgs_flip).squeeze(1)
            probs_flip = torch.sigmoid(logits_flip)

            # Average probabilities
            avg_probs = (probs_orig + probs_flip) / 2.0
            all_preds.extend(avg_probs.cpu().numpy())

    # Save Submission
    # Ensure length matches
    if len(all_preds) != len(df_test):
        print(
            f"Warning: Prediction count ({len(all_preds)}) does not match metadata count ({len(df_test)})."
        )

    df_test["target"] = all_preds

    # Select only required columns
    submission_df = df_test[["id", "target"]]
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
