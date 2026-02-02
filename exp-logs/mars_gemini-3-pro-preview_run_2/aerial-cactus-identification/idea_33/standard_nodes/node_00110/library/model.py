import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from library
from library.config import Config
from library.utils import (
    set_seed,
    calculate_roc_auc,
    save_checkpoint,
    load_checkpoint,
    save_submission,
)
from library.dataset import get_dataloaders
from library.model_components import RepNeXtBlock, RepDownsample

# -------------------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------------------


class WideSERepNeXt(nn.Module):
    """
    Custom Wide SE-RepNeXt Architecture with Multi-Scale Aggregation.

    Attributes:
        - Backbone: 3-stage RepNeXt with 'Super-Wide' channels [64, 128, 256].
        - Blocks: RepNeXtBlock (Grouped Conv + 1x1 + Identity) + SE.
        - Downsampling: RepDownsample (Parallel Strided 3x3 + 1x1).
        - Head: Multi-Scale Aggregation (GAP Stage 2 + GAP Stage 3 -> Concat -> Linear).
        - Inference: Supports structural re-parameterization.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, deploy=False):
        super(WideSERepNeXt, self).__init__()
        self.deploy = deploy

        # Configuration
        channels = Config.CHANNELS  # [64, 128, 256]
        groups = Config.CARDINALITY  # 32

        # 1. Stem
        # Standard 3x3 Conv to map input (3) to initial width (64).
        # We use a standard ConvBNReLU here as the entry point.
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # 2. Stage 1
        # Width: 64. Resolution: 32x32.
        # We use 2 blocks for the stage.
        self.stage1 = nn.Sequential(
            RepNeXtBlock(channels[0], channels[0], groups=groups, deploy=deploy),
            RepNeXtBlock(channels[0], channels[0], groups=groups, deploy=deploy),
        )

        # 3. Downsample 1
        # 64 -> 128. Resolution: 32x32 -> 16x16.
        self.downsample1 = RepDownsample(
            channels[0], channels[1], groups=groups, deploy=deploy
        )

        # 4. Stage 2
        # Width: 128. Resolution: 16x16.
        self.stage2 = nn.Sequential(
            RepNeXtBlock(channels[1], channels[1], groups=groups, deploy=deploy),
            RepNeXtBlock(channels[1], channels[1], groups=groups, deploy=deploy),
        )

        # 5. Downsample 2
        # 128 -> 256. Resolution: 16x16 -> 8x8.
        self.downsample2 = RepDownsample(
            channels[1], channels[2], groups=groups, deploy=deploy
        )

        # 6. Stage 3
        # Width: 256. Resolution: 8x8.
        self.stage3 = nn.Sequential(
            RepNeXtBlock(channels[2], channels[2], groups=groups, deploy=deploy),
            RepNeXtBlock(channels[2], channels[2], groups=groups, deploy=deploy),
        )

        # 7. Head (Multi-Scale Aggregation)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Input to FC is concatenation of Stage 2 (128) and Stage 3 (256) features
        fc_in_features = channels[1] + channels[2]
        self.fc = nn.Linear(fc_in_features, num_classes)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Stage 1
        x = self.stage1(x)

        # Transition to Stage 2
        x = self.downsample1(x)
        x = self.stage2(x)
        feat_s2 = x  # Save Stage 2 features (16x16)

        # Transition to Stage 3
        x = self.downsample2(x)
        x = self.stage3(x)
        feat_s3 = x  # Save Stage 3 features (8x8)

        # Multi-Scale Aggregation
        # GAP on Stage 2
        pool_s2 = self.gap(feat_s2).view(feat_s2.size(0), -1)
        # GAP on Stage 3
        pool_s3 = self.gap(feat_s3).view(feat_s3.size(0), -1)

        # Concatenate
        combined = torch.cat([pool_s2, pool_s3], dim=1)

        # Classifier
        out = self.fc(combined)
        return out

    def reparameterize(self):
        """
        Switches the model to inference mode by fusing branches in RepNeXt and RepDownsample blocks.
        """
        if self.deploy:
            return

        for module in self.modules():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()

        self.deploy = True


# -------------------------------------------------------------------------
# Training & Inference Logic
# -------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # BCEWithLogitsLoss expects (N, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store for metrics
        probs = torch.sigmoid(outputs)
        all_targets.append(labels.detach().cpu().numpy())
        all_preds.append(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            all_targets.append(labels.detach().cpu().numpy())
            all_preds.append(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def run_training(seed):
    """
    Trains a single model instance with the specified seed.
    """
    set_seed(seed)
    device = Config.DEVICE

    # Data
    train_loader, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    # Model
    model = WideSERepNeXt(deploy=False).to(device)

    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN)

    # Training Loop
    best_val_auc = 0.0
    patience_counter = 0

    print(f"\n[Seed {seed}] Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            save_checkpoint(model.state_dict(), seed)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"[Seed {seed}] Best Val AUC: {best_val_auc:.6f}")


def run_inference_and_submission():
    """
    Loads all trained models, performs inference with TTA, averages predictions, and saves submission.
    """
    print("\nStarting Inference with Ensemble and TTA...")
    device = Config.DEVICE
    _, _, test_loader, test_ids = get_dataloaders(load_cached_data=True)

    # Prepare to store predictions for each seed
    # Shape: (Num_Seeds, Num_Samples)
    ensemble_preds = []

    for seed in Config.SEEDS:
        print(f"Processing Seed {seed}...")

        # Load Model
        model = WideSERepNeXt(deploy=False).to(device)
        try:
            model = load_checkpoint(model, seed, device)
        except FileNotFoundError:
            print(f"Warning: Checkpoint for seed {seed} not found. Skipping.")
            continue

        # Reparameterize for inference speed
        model.reparameterize()
        model.eval()

        seed_preds = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)

                # Test Time Augmentation (TTA)
                # 1. Original
                out_orig = model(images)
                prob_orig = torch.sigmoid(out_orig)

                # 2. Horizontal Flip
                images_h = torch.flip(images, [3])
                out_h = model(images_h)
                prob_h = torch.sigmoid(out_h)

                # 3. Vertical Flip
                images_v = torch.flip(images, [2])
                out_v = model(images_v)
                prob_v = torch.sigmoid(out_v)

                # Average TTA probabilities
                avg_prob = (prob_orig + prob_h + prob_v) / 3.0
                seed_preds.append(avg_prob.cpu().numpy())

        seed_preds = np.concatenate(seed_preds).flatten()
        ensemble_preds.append(seed_preds)

    if not ensemble_preds:
        print("Error: No predictions generated.")
        return

    # Average across seeds (Homogeneous Seed Averaging)
    ensemble_preds = np.array(ensemble_preds)
    final_preds = np.mean(ensemble_preds, axis=0)

    # Save Submission
    save_submission(test_ids, final_preds)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    """
    Main execution pipeline.
    """
    # 1. Train 5 independent instances
    for seed in Config.SEEDS:
        run_training(seed)

    # 2. Generate submission
    run_inference_and_submission()
