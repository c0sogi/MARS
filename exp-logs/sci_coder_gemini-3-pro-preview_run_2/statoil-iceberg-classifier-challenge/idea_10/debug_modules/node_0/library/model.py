import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger, AverageMeter, save_checkpoint
from library.data_loader import get_kfold_loaders, get_test_loader

# ==========================================
# 1. ATTENTION MODULES (CBAM)
# ==========================================


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ==========================================
# 2. MODEL ARCHITECTURE (MLCW-Net)
# ==========================================


class MLCWNet(nn.Module):
    def __init__(self):
        super(MLCWNet, self).__init__()

        # Configuration
        filters = Config.BACKBONE_FILTERS  # [32, 48, 64, 32]

        # --- Stage 1 ---
        self.conv1 = nn.Conv2d(3, filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.cbam1 = CBAM(filters[0])

        # --- Stage 2 ---
        self.conv2 = nn.Conv2d(
            filters[0], filters[1], kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.cbam2 = CBAM(filters[1])

        # --- Stage 3 (Level A Source) ---
        self.conv3 = nn.Conv2d(
            filters[1], filters[2], kernel_size=3, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.cbam3 = CBAM(filters[2])

        # --- Stage 4 (Level B Source) ---
        self.conv4 = nn.Conv2d(
            filters[2], filters[3], kernel_size=3, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(filters[3])
        self.cbam4 = CBAM(filters[3])

        # Pooling Layers
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pooling for Level A feature extraction to ensure 4x4 grid
        self.adaptive_pool_a = nn.AdaptiveMaxPool2d(
            (Config.FUSION_POOL_SIZE, Config.FUSION_POOL_SIZE)
        )

        # --- Metadata Branch ---
        self.meta_bn = nn.BatchNorm1d(1)
        self.meta_fc1 = nn.Linear(1, 16)
        self.meta_fc2 = nn.Linear(16, 32)

        # --- Fusion Head ---
        # Calculate Flattened Sizes
        # Level A: filters[2] (64) * 4 * 4 = 1024
        # Level B: filters[3] (32) * 4 * 4 = 512 (Assuming input 75x75 -> 37 -> 18 -> 9 -> 4)
        self.dim_a = filters[2] * Config.FUSION_POOL_SIZE * Config.FUSION_POOL_SIZE
        self.dim_b = filters[3] * Config.FUSION_POOL_SIZE * Config.FUSION_POOL_SIZE
        self.dim_meta = 32

        total_dim = self.dim_a + self.dim_b + self.dim_meta

        self.head_fc1 = nn.Linear(total_dim, 512)
        self.head_bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.head_fc2 = nn.Linear(512, 1)

    def forward(self, x, inc_angle):
        # --- Stage 1 ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.cbam1(x)
        x = self.maxpool(x)  # 75 -> 37

        # --- Stage 2 ---
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.cbam2(x)
        x = self.maxpool(x)  # 37 -> 18

        # --- Stage 3 ---
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.cbam3(x)

        # Level A Extraction (Mid-Level)
        feat_a = self.adaptive_pool_a(x)  # Force to 4x4
        feat_a = torch.flatten(feat_a, 1)

        x = self.maxpool(x)  # 18 -> 9

        # --- Stage 4 ---
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.cbam4(x)
        x = self.maxpool(x)  # 9 -> 4

        # Level B Extraction (High-Level)
        feat_b = torch.flatten(x, 1)

        # --- Metadata Branch ---
        inc_angle = inc_angle.view(-1, 1)  # Ensure (N, 1)
        m = self.meta_bn(inc_angle)
        m = F.relu(self.meta_fc1(m))
        m = F.relu(self.meta_fc2(m))

        # --- Fusion ---
        combined = torch.cat([feat_a, feat_b, m], dim=1)

        out = self.head_fc1(combined)
        out = self.head_bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.head_fc2(out)

        return out


# ==========================================
# 3. TRAINING LOGIC
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()

    for images, angles, targets in loader:
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for images, angles, targets in loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

    return losses.avg


def train_model():
    """
    Executes the Stratified 5-Fold Cross-Validation training loop.
    Saves the best model for each fold.
    """
    logger = setup_logger(
        "MLCWNet_Training", os.path.join(Config.WORKING_DIR, "train.log")
    )
    device = torch.device(Config.DEVICE)

    logger.info(f"Starting training on device: {device}")

    # Get K-Fold Loaders
    fold_loaders = get_kfold_loaders(load_cached_data=True)

    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        logger.info(f"\n{'='*20} Fold {fold+1}/{Config.NUM_FOLDS} {'='*20}")

        # Initialize Model
        model = MLCWNet().to(device)

        # Optimization
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=True,
        )

        # Early Stopping State
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss = validate(model, val_loader, criterion, device)

            # Scheduler Step
            scheduler.step(val_loss)

            # Logging
            logger.info(
                f"Fold {fold+1} Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss:.16f} - Val Loss: {val_loss:.16f}"
            )

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                # Optional: Save intermediate best
                # save_checkpoint(best_model_state, os.path.join(Config.WORKING_DIR, f"fold_{fold}_best.pth"))
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Save best model for this fold
        save_path = os.path.join(Config.WORKING_DIR, f"mlcw_net_fold_{fold}.pth")
        if best_model_state is not None:
            save_checkpoint(best_model_state, save_path)
            logger.info(
                f"Saved best model for fold {fold+1} to {save_path} (Val Loss: {best_val_loss:.6f})"
            )
        else:
            logger.warning(f"No best model saved for fold {fold+1}!")


# ==========================================
# 4. INFERENCE LOGIC
# ==========================================


def generate_submission():
    """
    Loads trained models from all folds, predicts on test set,
    ensembles predictions, and saves submission file.
    """
    logger = setup_logger(
        "MLCWNet_Inference", os.path.join(Config.WORKING_DIR, "inference.log")
    )
    device = torch.device(Config.DEVICE)

    # Load Test Data
    test_loader, test_ids = get_test_loader(load_cached_data=True)

    # Storage for predictions
    # Shape: (Num_Test_Samples, Num_Folds)
    all_preds = []

    logger.info("Starting inference...")

    for fold in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"mlcw_net_fold_{fold}.pth")

        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}. Skipping fold.")
            continue

        # Load Model
        model = MLCWNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_preds = []

        with torch.no_grad():
            for images, angles, _ in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                # Forward pass
                logits = model(images, angles)
                probs = torch.sigmoid(logits)

                fold_preds.extend(probs.cpu().numpy().flatten())

        all_preds.append(fold_preds)
        logger.info(f"Fold {fold+1} inference complete.")

    if not all_preds:
        logger.error("No predictions generated. Check if models were trained.")
        return

    # Ensemble: Average Probabilities
    all_preds = np.array(all_preds)  # (Folds, N)
    avg_preds = np.mean(all_preds, axis=0)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

    # Save
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission_df.to_csv(save_path, index=False)
    logger.info(f"Submission saved to {save_path}")
