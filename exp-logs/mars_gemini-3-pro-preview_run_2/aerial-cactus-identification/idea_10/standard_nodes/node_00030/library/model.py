import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import (
    CHANNEL_CONFIG,
    USE_SE_BLOCK,
    USE_MULTI_SCALE,
    DROPOUT_RATE,
    SEEDS,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    MODEL_DIR,
    set_seed,
)
from library.utils import (
    get_device,
    calculate_roc_auc,
    AverageMeter,
    save_checkpoint,
    save_submission,
)
from library.dataset import get_dataloaders


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResidualBlock, self).__init__()
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
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CustomNarrowSEMultiScaleResNet(nn.Module):
    def __init__(self):
        super(CustomNarrowSEMultiScaleResNet, self).__init__()
        self.use_multi_scale = USE_MULTI_SCALE
        c1, c2, c3 = CHANNEL_CONFIG

        # Stem: 3x3 conv, no aggressive downsampling yet
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32x32 resolution
        self.stage1 = self._make_layer(c1, c1, stride=1)

        # Stage 2: 16x16 resolution
        self.stage2 = self._make_layer(c1, c2, stride=2)

        # Stage 3: 8x8 resolution
        self.stage3 = self._make_layer(c2, c3, stride=2)

        # Classifier Head
        # If multi-scale, we concat GAP(Stage2) and GAP(Stage3)
        fc_in_features = c3
        if self.use_multi_scale:
            fc_in_features = c2 + c3

        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.fc = nn.Linear(fc_in_features, 1)

    def _make_layer(self, in_channels, out_channels, stride, blocks=2):
        layers = []
        layers.append(
            ResidualBlock(in_channels, out_channels, stride, use_se=USE_SE_BLOCK)
        )
        for _ in range(1, blocks):
            layers.append(
                ResidualBlock(out_channels, out_channels, 1, use_se=USE_SE_BLOCK)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)

        x = self.stage2(x)
        feat2 = x

        x = self.stage3(x)
        feat3 = x

        # Aggregation
        f3 = F.adaptive_avg_pool2d(feat3, 1).flatten(1)

        if self.use_multi_scale:
            f2 = F.adaptive_avg_pool2d(feat2, 1).flatten(1)
            combined = torch.cat([f2, f3], dim=1)
        else:
            combined = f3

        combined = self.dropout(combined)
        out = self.fc(combined)
        return out


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()
    scores = []
    targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

        # Store for AUC calculation
        scores.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())

    auc = calculate_roc_auc(targets, scores)
    return losses.avg, auc


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    scores = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))
            scores.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    auc = calculate_roc_auc(targets, scores)
    return losses.avg, auc


def train_model(seed):
    set_seed(seed)
    device = get_device()

    # DataLoaders
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=4, load_cached_data=True
    )

    # Model
    model = CustomNarrowSEMultiScaleResNet().to(device)

    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"\nStarting training for Seed {seed}...")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save best model immediately
            save_checkpoint(best_model_state, f"model_seed_{seed}.pth")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Seed {seed} completed. Best Val AUC: {best_val_auc:.6f}")
    return best_val_auc


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================


def predict_with_tta(model, images, device):
    """
    Predicts using Original, H-Flip, and V-Flip.
    Returns average probability.
    """
    model.eval()
    with torch.no_grad():
        # 1. Original
        out_orig = torch.sigmoid(model(images))

        # 2. Horizontal Flip
        images_h = torch.flip(images, [3])
        out_h = torch.sigmoid(model(images_h))

        # 3. Vertical Flip
        images_v = torch.flip(images, [2])
        out_v = torch.sigmoid(model(images_v))

        # Average
        avg_pred = (out_orig + out_h + out_v) / 3.0

    return avg_pred


def generate_submission():
    device = get_device()
    _, _, test_loader, test_ids = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=4, load_cached_data=True
    )

    # Iterate over seeds
    models = []
    for seed in SEEDS:
        model_path = os.path.join(MODEL_DIR, f"model_seed_{seed}.pth")
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        model = CustomNarrowSEMultiScaleResNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)

    if not models:
        print("No models found for inference.")
        return

    print(f"\nGenerating predictions using {len(models)} models with TTA...")

    final_probs = []

    # Iterate through test loader
    # We process batch by batch for all models to save memory
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            batch_preds_sum = torch.zeros((images.size(0), 1), device=device)

            for model in models:
                # Get TTA averaged prediction for this model
                preds = predict_with_tta(model, images, device)
                batch_preds_sum += preds

            # Average over models
            batch_preds_avg = batch_preds_sum / len(models)
            final_probs.extend(batch_preds_avg.cpu().numpy().flatten())

    save_submission(test_ids, final_probs, "submission.csv")
    print("Submission saved to submission.csv")


def main():
    # 1. Train models for all seeds
    val_aucs = []
    for seed in SEEDS:
        auc = train_model(seed)
        val_aucs.append(auc)

    print(f"\nTraining Complete. Average Best Val AUC: {np.mean(val_aucs):.6f}")

    # 2. Generate Submission
    generate_submission()
