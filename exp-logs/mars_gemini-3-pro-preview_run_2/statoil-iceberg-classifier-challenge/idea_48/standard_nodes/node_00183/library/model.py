import os
import json
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# Import from provided library files
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    SUBMISSION_DIR,
    TRAIN_JSON,
    TEST_JSON,
    TRAIN_META_PATH,
    CACHE_FILE,
    SUBMISSION_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    DROPOUT_RATE,
    NUM_FOLDS,
    DEVICE,
    SEED,
)
from library.utils import seed_everything, get_logger
from library.layers import WideConvBlock, CBAM, DualPooling

# Set up logger
logger = get_logger("model", os.path.join(WORKING_DIR, "model.log"))

# ==========================================
# 1. DATA PROCESSING & DATASET
# ==========================================


def get_processed_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into 3-channel images, applies global normalization,
    and caches the result.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(CACHE_FILE):
        logger.info(f"Loading cached data from {CACHE_FILE}")
        try:
            data = np.load(CACHE_FILE, allow_pickle=True)
            return (
                data["X_train"],
                data["y_train"],
                data["inc_train"],
                data["X_test"],
                data["inc_test"],
                data["test_ids"],
            )
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing...")

    logger.info("Processing data from scratch...")

    # Load Train Data
    with open(TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    df_train = pd.DataFrame(train_data)

    # Load Test Data
    with open(TEST_JSON, "r") as f:
        test_data = json.load(f)
    df_test = pd.DataFrame(test_data)

    # Helper to process bands
    def process_images(df):
        b1 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
        )
        b2 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
        )
        # Channel 3: Average of Band 1 and Band 2
        b3 = (b1 + b2) / 2.0

        # Stack: (N, 3, 75, 75)
        images = np.stack([b1, b2, b3], axis=1)
        return images

    X_train = process_images(df_train)
    X_test = process_images(df_test)

    # Extract Targets and Metadata
    y_train = df_train["is_iceberg"].values.astype(np.int64)

    # Handle Incidence Angle (Replace 'na' with 0.0 for now, will be masked or handled by network?)
    # Instructions say 'na' are only in train. Standard practice: fill with mean or 0.
    # We will use fill with 0 and let the network handle it, or fill with mean.
    # Let's use mean imputation for simplicity as per standard baselines.

    def process_inc_angle(series):
        # Coerce to numeric, turning 'na' to NaN
        vals = pd.to_numeric(series, errors="coerce")
        # Fill NaN with 0.0 (or mean). Using 0.0 as a safe default if mean is problematic,
        # but mean is better for distribution.
        mask = vals.isna()
        if mask.sum() > 0:
            vals[mask] = vals.mean()  # Simple mean imputation
        return vals.values.astype(np.float32)

    inc_train = process_inc_angle(df_train["inc_angle"])
    inc_test = process_inc_angle(df_test["inc_angle"])
    test_ids = df_test["id"].values

    # Global Min-Max Scaling
    # Compute stats on TRAIN only
    logger.info("Computing global scaling statistics...")

    # Reshape to (N*H*W, C) for stats
    train_pixels = X_train.transpose(0, 2, 3, 1).reshape(-1, 3)

    min_vals = train_pixels.min(axis=0)
    max_vals = train_pixels.max(axis=0)

    logger.info(f"Global Min: {min_vals}, Global Max: {max_vals}")

    # Apply scaling: (X - min) / (max - min)
    # Allow values > 1.0 or < 0.0 in test set (No Hard Clipping)

    for c in range(3):
        denom = max_vals[c] - min_vals[c] + 1e-8
        X_train[:, c, :, :] = (X_train[:, c, :, :] - min_vals[c]) / denom
        X_test[:, c, :, :] = (X_test[:, c, :, :] - min_vals[c]) / denom

    # Save to cache
    logger.info(f"Saving processed data to {CACHE_FILE}")
    np.savez(
        CACHE_FILE,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train,
        X_test=X_test,
        inc_test=inc_test,
        test_ids=test_ids,
    )

    return X_train, y_train, inc_train, X_test, inc_test, test_ids


class IcebergDataset(Dataset):
    def __init__(self, X, inc_angles, y=None, transform=False):
        self.X = X
        self.inc_angles = inc_angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]  # (3, 75, 75)
        inc = self.inc_angles[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()

        if self.transform:
            # Augmentation: Random Rotation (0, 90, 180, 270)
            k = random.randint(0, 3)
            img_tensor = torch.rot90(img_tensor, k, [1, 2])

            # Augmentation: Horizontal Flip
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])  # [C, H, W], flip W

        inc_tensor = torch.tensor([inc], dtype=torch.float32)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.long)
            return img_tensor, inc_tensor, label
        else:
            return img_tensor, inc_tensor


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================


class RobustDualPathReadout(nn.Module):
    """
    Robust Dual-Path Readout (Cite solution_lesson_node_00123).

    Decouples spatial structure from global statistics:
    Path A (Spatial): 3x3 Convolution -> Flatten -> BN (Cite solution_lesson_node_00182: Avoid dilation on small maps).
    Path B (Intensity): Global Average Pooling -> BN.

    Applies Late-Stage Branch Normalization (Cite solution_lesson_node_00161).
    """

    def __init__(self, in_channels, spatial_out_channels=48):
        super(RobustDualPathReadout, self).__init__()

        # Path A: Spatial Features (Structure)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, spatial_out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
        )
        # Flattened size: spatial_out_channels * 4 * 4
        self.flat_dim = spatial_out_channels * 4 * 4
        self.bn_spatial = nn.BatchNorm1d(self.flat_dim)

        # Path B: Robust Intensity (Invariance)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.bn_intensity = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        # x: (B, 256, 4, 4)

        # Path A
        s = self.spatial_conv(x)
        s = s.view(s.size(0), -1)
        s = self.bn_spatial(s)

        # Path B
        i = self.global_pool(x).view(x.size(0), -1)
        i = self.bn_intensity(i)

        # Fuse
        return torch.cat([s, i], dim=1)


class RDPWBN(nn.Module):
    """
    Robust Dual-Path Wide-Body Network.

    Replaces Dilated Readout with Robust Dual-Path Readout.
    """

    def __init__(self):
        super(RDPWBN, self).__init__()

        # --- Visual Branch ---
        # Stage 1
        self.stage1_conv = WideConvBlock(3, 128)
        self.stage1_cbam = CBAM(128)
        self.stage1_pool = DualPooling(2, 2)  # Out: 256 channels

        # Stage 2
        self.stage2_conv = WideConvBlock(256, 128)
        self.stage2_cbam = CBAM(128)
        self.stage2_pool = DualPooling(2, 2)  # Out: 256 channels

        # Stage 3
        self.stage3_conv = WideConvBlock(256, 128)
        self.stage3_cbam = CBAM(128)
        self.stage3_pool = DualPooling(2, 2)  # Out: 256 channels

        # Stage 4
        self.stage4_conv = WideConvBlock(256, 128)
        self.stage4_cbam = CBAM(128)
        self.stage4_pool = DualPooling(2, 2)  # Out: 256 channels, 4x4 spatial

        # Readout
        # Spatial: 48 * 16 = 768. Intensity: 256. Total: 1024.
        self.readout = RobustDualPathReadout(256, spatial_out_channels=48)

        # --- Metadata Branch ---
        # Cite solution_lesson_node_00161: Normalize branch before fusion
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            # No final ReLU to allow centered distribution matching BN outputs of visual branches
        )

        # --- Fusion Head ---
        # Input: 1024 (Visual) + 32 (Meta) = 1056
        self.head = nn.Sequential(
            nn.Linear(1056, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 2),  # Logits
        )

    def forward(self, img, inc_angle):
        # Visual Path
        x = self.stage1_conv(img)
        x = self.stage1_cbam(x)
        x = self.stage1_pool(x)

        x = self.stage2_conv(x)
        x = self.stage2_cbam(x)
        x = self.stage2_pool(x)

        x = self.stage3_conv(x)
        x = self.stage3_cbam(x)
        x = self.stage3_pool(x)

        x = self.stage4_conv(x)
        x = self.stage4_cbam(x)
        x = self.stage4_pool(x)

        visual_feat = self.readout(x)

        # Metadata Path
        meta_feat = self.meta_mlp(inc_angle)

        # Fusion
        combined = torch.cat([visual_feat, meta_feat], dim=1)
        logits = self.head(combined)

        return logits


# ==========================================
# 3. TRAINING & INFERENCE PIPELINE
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, inc_angles, labels in loader:
        images = images.to(device)
        inc_angles = inc_angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, inc_angles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    preds = []
    targets = []

    with torch.no_grad():
        for images, inc_angles, labels in loader:
            images = images.to(device)
            inc_angles = inc_angles.to(device)
            labels = labels.to(device)

            outputs = model(images, inc_angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            preds.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.array(preds), np.array(targets)


def run_training_and_inference():
    """
    Main execution function.
    1. Loads data.
    2. Runs Stratified 5-Fold CV.
    3. Trains model with Early Stopping.
    4. Generates predictions on Test set.
    5. Saves submission.
    """
    seed_everything(SEED)

    # 1. Load Data
    X, y, inc, X_test, inc_test, test_ids = get_processed_data()

    # Test Dataset (No labels)
    test_dataset = IcebergDataset(X_test, inc_test, y=None, transform=False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Placeholders for OOF and Test Predictions
    oof_preds = np.zeros(len(X))
    test_preds_accum = np.zeros(len(X_test))

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n{'='*20} Fold {fold+1}/{NUM_FOLDS} {'='*20}")

        # Split Data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        inc_train_fold, inc_val_fold = inc[train_idx], inc[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(
            X_train_fold, inc_train_fold, y_train_fold, transform=True
        )
        val_ds = IcebergDataset(X_val_fold, inc_val_fold, y_val_fold, transform=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
        )

        # Initialize Model
        model = DCWBN().to(DEVICE)

        # Optimizer & Loss
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Training Loop
        best_val_loss = float("inf")
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss, val_acc, val_probs, _ = validate(
                model, val_loader, criterion, DEVICE
            )

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
                logger.info(
                    f"Epoch {epoch+1}: Train Loss {train_loss:.5f} | Val Loss {val_loss:.5f} | Val Acc {val_acc:.4f} *"
                )
            else:
                patience_counter += 1
                if epoch % 5 == 0:
                    logger.info(
                        f"Epoch {epoch+1}: Train Loss {train_loss:.5f} | Val Loss {val_loss:.5f} | Val Acc {val_acc:.4f}"
                    )

            if patience_counter >= PATIENCE:
                logger.info("Early stopping triggered")
                break

        # Load Best Model
        model.load_state_dict(best_model_wts)

        # Save Model
        model_path = os.path.join(WORKING_DIR, f"model_fold_{fold}.pth")
        torch.save(model.state_dict(), model_path)

        # Generate OOF Predictions
        _, _, val_probs, _ = validate(model, val_loader, criterion, DEVICE)
        oof_preds[val_idx] = val_probs

        # Generate Test Predictions (Fold Contribution)
        model.eval()
        fold_test_preds = []
        with torch.no_grad():
            for images, inc_angles in test_loader:
                images = images.to(DEVICE)
                inc_angles = inc_angles.to(DEVICE)
                outputs = model(images, inc_angles)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                fold_test_preds.extend(probs.cpu().numpy())

        test_preds_accum += np.array(fold_test_preds)

    # Average Test Predictions
    avg_test_preds = test_preds_accum / NUM_FOLDS

    # Create Submission
    submission = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds})

    submission.to_csv(SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {SUBMISSION_PATH}")

    # Log OOF Metric
    from sklearn.metrics import log_loss

    oof_loss = log_loss(y, oof_preds)
    logger.info(f"Overall OOF Log Loss: {oof_loss:.6f}")
