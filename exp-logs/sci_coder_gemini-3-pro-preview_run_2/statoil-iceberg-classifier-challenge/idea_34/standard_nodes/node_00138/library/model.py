import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import copy

# Import configuration and utilities
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    SUBMISSION_DIR,
    TRAIN_JSON,
    TEST_JSON,
    CACHE_PATH,
    SUBMISSION_PATH,
    get_model_path,
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    DROPOUT_RATE,
    NUM_FOLDS,
    PATIENCE,
    IMAGE_SIZE,
    NUM_CHANNELS,
    GEM_P_INIT,
    DEVICE,
    set_seed,
)
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
)

# ==========================================
# 1. LAYERS & MODEL ARCHITECTURE
# ==========================================


class GeM(nn.Module):
    """
    Generalized Mean Pooling.
    Learns to interpolate between Average Pooling (p=1) and Max Pooling (p=inf).
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (N, C, H, W) -> (N, C)
        # clamp to avoid nan
        return (
            F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1)))
            .pow(1.0 / p)
            .squeeze(-1)
            .squeeze(-1)
        )


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Uses Mixed Pooling (Max + Avg) for both Channel and Spatial attention.
    """

    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channels = channels

        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(
            2, 1, kernel_size=7, stride=1, padding=3, bias=False
        )

    def forward(self, x):
        # Channel Attention
        b, c, _, _ = x.size()
        # Avg Pool
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3))).view(b, c)
        channel_avg = self.mlp(avg_pool)
        # Max Pool
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3))).view(b, c)
        channel_max = self.mlp(max_pool)

        channel_scale = torch.sigmoid(channel_avg + channel_max).view(b, c, 1, 1)
        x = x * channel_scale

        # Spatial Attention
        # Avg along channels
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        # Max along channels
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([spatial_avg, spatial_max], dim=1)

        spatial_scale = torch.sigmoid(self.conv_spatial(spatial_in))
        x = x * spatial_scale

        return x


class WideBodyBlock(nn.Module):
    """
    Wide-Body Block with Delayed Integration and Dual-Stream Pooling.
    """

    def __init__(self, in_channels, out_channels):
        super(WideBodyBlock, self).__init__()

        # Wide Convolution: Maps input to 'out_channels' (e.g., 256 -> 128)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Attention
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)

        # Dual-Stream Pooling: Max + Min
        # Min pooling is implemented as -Max(-x)
        pool_max = F.max_pool2d(x, kernel_size=2, stride=2)
        pool_min = -F.max_pool2d(-x, kernel_size=2, stride=2)

        # Concatenate: out_channels * 2
        return torch.cat([pool_max, pool_min], dim=1)


class GA_WBN(nn.Module):
    """
    GeM-Augmented Wide-Body Network.
    """

    def __init__(self):
        super(GA_WBN, self).__init__()

        # --- Visual Branch ---
        # Stage 1: Standard Entry
        self.stage1_conv = nn.Conv2d(
            NUM_CHANNELS, 64, kernel_size=3, padding=1, bias=False
        )
        self.stage1_bn = nn.BatchNorm2d(64)
        self.stage1_pool = nn.MaxPool2d(2, 2)  # 75 -> 37

        # Stage 2: WideBody 64 -> 128 (DualPool -> 256)
        self.stage2 = WideBodyBlock(64, 128)  # 37 -> 18

        # Stage 3: WideBody 256 -> 128 (DualPool -> 256)
        self.stage3 = WideBodyBlock(256, 128)  # 18 -> 9

        # Stage 4: WideBody 256 -> 128 (DualPool -> 256)
        self.stage4 = WideBodyBlock(256, 128)  # 9 -> 4

        # Readout Path A: Spatial Context
        # Input 256 channels, 4x4 spatial
        self.readout_conv = nn.Conv2d(256, 48, kernel_size=3, padding=1)

        # Readout Path B: Intensity
        self.readout_gem = GeM(p=GEM_P_INIT)

        # --- Metadata Branch ---
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # --- Fusion Head ---
        # Path A (4*4*48 = 768) + Path B (256) + Meta (32) = 1056
        self.fusion_dim = 768 + 256 + 32

        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 1),
        )

    def forward(self, x_img, x_angle):
        # Visual
        x = F.relu(self.stage1_bn(self.stage1_conv(x_img)))
        x = self.stage1_pool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Path A
        ctx = self.readout_conv(x)
        ctx = ctx.view(ctx.size(0), -1)  # Flatten

        # Path B
        inte = self.readout_gem(x)  # Flattened by GeM

        # Visual Vector
        vis_vec = torch.cat([ctx, inte], dim=1)

        # Metadata
        meta_vec = self.meta_mlp(x_angle)

        # Fusion
        combined = torch.cat([vis_vec, meta_vec], dim=1)
        out = self.head(combined)

        return out


# ==========================================
# 2. DATA HANDLING
# ==========================================


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=False):
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image: (3, 75, 75)
        img = self.images[idx]
        angle = self.angles[idx]

        # Augmentation
        if self.transform:
            # Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            img = np.rot90(img, k, axes=(1, 2))

            # Random Horizontal Flip
            if np.random.random() > 0.5:
                img = np.flip(img, axis=2)  # axis 2 is width

            # No Vertical Flip (axis 1) as per instructions

        # Convert to tensor
        img_tensor = torch.from_numpy(img.copy()).float()
        angle_tensor = torch.tensor([angle], dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor([self.labels[idx]], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor


def process_and_cache_data(load_cached_data=True):
    """
    Loads JSON data, processes it into tensors, normalizes globally, and caches it.
    """
    if load_cached_data and os.path.exists(CACHE_PATH):
        print(f"Loading cached data from {CACHE_PATH}...")
        data = np.load(CACHE_PATH)
        return (
            data["X_train"],
            data["y_train"],
            data["inc_train"],
            data["X_test"],
            data["inc_test"],
            data["ids_test"],
        )

    print("Processing data from scratch...")

    # Load JSONs
    with open(TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    with open(TEST_JSON, "r") as f:
        test_data = json.load(f)

    # Helper to process list of dicts
    def process_json(data_list, is_train=True):
        images = []
        angles = []
        ids = []
        labels = []

        # Calculate mean angle from training data for imputation
        # We do this in a first pass or just gather valid ones
        valid_angles = [x["inc_angle"] for x in data_list if x["inc_angle"] != "na"]
        mean_angle = np.mean(valid_angles) if valid_angles else 0.0

        for item in data_list:
            # Bands
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            # Channel 3: Mean
            b3 = (b1 + b2) / 2.0

            # Stack: (3, 75, 75)
            img = np.stack([b1, b2, b3], axis=0)
            images.append(img)

            # Angle
            ang = item["inc_angle"]
            if ang == "na":
                ang = mean_angle
            angles.append(float(ang))

            ids.append(item["id"])

            if is_train:
                labels.append(item["is_iceberg"])

        return (
            np.array(images),
            np.array(angles),
            np.array(ids),
            np.array(labels) if is_train else None,
        )

    # Process Train
    print("Parsing train.json...")
    X_train_raw, inc_train_raw, _, y_train = process_json(train_data, is_train=True)

    # Process Test
    print("Parsing test.json...")
    X_test_raw, inc_test_raw, ids_test, _ = process_json(test_data, is_train=False)

    # Global Normalization
    # Compute stats on Training set ONLY
    print("Computing global stats...")
    # Flatten to (N*H*W, C) or similar to get global min/max per channel
    # X_train_raw shape: (N, 3, 75, 75)

    mins = []
    maxs = []
    for c in range(3):
        channel_data = X_train_raw[:, c, :, :]
        mins.append(np.min(channel_data))
        maxs.append(np.max(channel_data))

    print(f"Global Mins: {mins}")
    print(f"Global Maxs: {maxs}")

    # Apply normalization
    def normalize(X, mins, maxs):
        X_norm = np.zeros_like(X)
        for c in range(3):
            # Min-Max Scaling
            # Allow values > 1.0 or < 0.0 for outliers in test/val
            X_norm[:, c, :, :] = (X[:, c, :, :] - mins[c]) / (maxs[c] - mins[c])
        return X_norm

    X_train = normalize(X_train_raw, mins, maxs)
    X_test = normalize(X_test_raw, mins, maxs)

    # Cache
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez(
        CACHE_PATH,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train_raw,
        X_test=X_test,
        inc_test=inc_test_raw,
        ids_test=ids_test,
    )

    print("Data processed and cached.")
    return X_train, y_train, inc_train_raw, X_test, inc_test_raw, ids_test


# ==========================================
# 3. TRAINING & EVALUATION
# ==========================================


def train_one_fold(fold_idx, train_idx, val_idx, X, y, inc, device):
    print(f"\n--- Fold {fold_idx + 1}/{NUM_FOLDS} ---")

    # Split Data
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    inc_tr, inc_val = inc[train_idx], inc[val_idx]

    # Datasets
    train_ds = IcebergDataset(X_tr, inc_tr, y_tr, transform=True)
    val_ds = IcebergDataset(X_val, inc_val, y_val, transform=False)

    # Loaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    # Model
    model = GA_WBN().to(device)

    # Optimizer (Adam, not AdamW)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping
    model_path = get_model_path(fold_idx)
    early_stopping = EarlyStopping(patience=PATIENCE, mode="min", verbose=True)

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, angs, lbls in train_loader:
            imgs, angs, lbls = imgs.to(device), angs.to(device), lbls.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, angs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, angs, lbls in val_loader:
                imgs, angs, lbls = imgs.to(device), angs.to(device), lbls.to(device)
                outputs = model(imgs, angs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_ds)

        # Logging
        # Print full precision
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Scheduler Step
        scheduler.step(val_loss)

        # Early Stopping
        early_stopping(val_loss, model, model_path)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    return early_stopping.best_score


def run_training():
    seed_everything(SEED)

    # Data
    X_train, y_train, inc_train, _, _, _ = process_and_cache_data(load_cached_data=True)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        best_loss = train_one_fold(
            fold_idx, train_idx, val_idx, X_train, y_train, inc_train, DEVICE
        )
        scores.append(best_loss)

    print("\n=========================")
    print(f"CV Average Log Loss: {np.mean(scores):.6f}")
    print("=========================")


def generate_submission():
    seed_everything(SEED)

    # Load Data
    _, _, _, X_test, inc_test, ids_test = process_and_cache_data(load_cached_data=True)

    test_ds = IcebergDataset(X_test, inc_test, labels=None, transform=False)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    # Ensemble Prediction
    fold_preds = []

    for fold_idx in range(NUM_FOLDS):
        model_path = get_model_path(fold_idx)
        model = GA_WBN().to(DEVICE)
        model = load_checkpoint(model, model_path, device=DEVICE)
        model.eval()

        preds = []
        with torch.no_grad():
            for imgs, angs in test_loader:
                imgs, angs = imgs.to(DEVICE), angs.to(DEVICE)
                outputs = model(imgs, angs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds.append(probs)

        fold_preds.append(np.concatenate(preds))

    # Average Predictions
    avg_preds = np.mean(fold_preds, axis=0).flatten()

    # Create Submission
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


# ==========================================
# MAIN ENTRY POINT
# ==========================================
def run_pipeline():
    print("Starting GA-WBN Pipeline...")
    run_training()
    generate_submission()
    print("Pipeline completed successfully.")
