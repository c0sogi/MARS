import os
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    CACHE_DIR,
    SUBMISSION_DIR,
    SUBMISSION_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INPUT_CHANNELS,
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_FOLDS,
    NUM_EPOCHS,
    PATIENCE,
    DROPOUT_RATE,
    BACKBONE_FILTERS,
    DEVICE,
    NUM_WORKERS,
    set_seed,
    TRAIN_JSON,
    TEST_JSON,
    TRAIN_META_PATH,
    TEST_META_PATH,
)
from library.utils import EarlyStopping

# ==========================================
# 1. DATA PROCESSING & CACHING
# ==========================================


def load_and_process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes images into 3 channels (HH, HV, Avg),
    handles missing incidence angles, applies global normalization,
    and caches the result.
    """
    cache_path = os.path.join(CACHE_DIR, "processed_data.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["X_train"],
            data["y_train"],
            data["inc_train"],
            data["X_test"],
            data["inc_test"],
            data["ids_test"],
        )

    print("Processing data from scratch...")

    # Load Metadata for reference (IDs)
    df_test_meta = pd.read_csv(TEST_META_PATH)

    # Load Raw JSONs
    with open(TRAIN_JSON, "r") as f:
        train_json = json.load(f)
    with open(TEST_JSON, "r") as f:
        test_json = json.load(f)

    # Map ID to data for O(1) access
    train_dict = {item["id"]: item for item in train_json}
    test_dict = {item["id"]: item for item in test_json}

    def process_images(ids, source_dict):
        images = []
        inc_angles = []

        for img_id in ids:
            item = source_dict[img_id]

            # Bands: Flattened list -> 75x75
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            # Channel 3: Average
            avg = (b1 + b2) / 2.0

            # Stack: (75, 75, 3)
            img = np.stack([b1, b2, avg], axis=-1)
            images.append(img)

            # Incidence Angle
            inc = item["inc_angle"]
            if inc == "na":
                inc_angles.append(np.nan)
            else:
                inc_angles.append(float(inc))

        return np.array(images, dtype=np.float32), np.array(
            inc_angles, dtype=np.float32
        )

    # Extract Train Data
    train_ids = [item["id"] for item in train_json]
    X_train, inc_train = process_images(train_ids, train_dict)
    y_train = np.array(
        [train_dict[i]["is_iceberg"] for i in train_ids], dtype=np.float32
    )

    # Extract Test Data
    test_ids = df_test_meta["id"].values
    X_test, inc_test = process_images(test_ids, test_dict)

    # Global Normalization
    # Compute min/max per channel across the ENTIRE training set
    stats = []
    for c in range(3):
        c_min = X_train[:, :, :, c].min()
        c_max = X_train[:, :, :, c].max()
        stats.append((c_min, c_max))

    def normalize(X, stats):
        X_norm = np.zeros_like(X, dtype=np.float32)
        for c in range(3):
            c_min, c_max = stats[c]
            denom = c_max - c_min if c_max != c_min else 1.0
            X_norm[:, :, :, c] = (X[:, :, :, c] - c_min) / denom
        return X_norm

    X_train = normalize(X_train, stats)
    X_test = normalize(X_test, stats)

    # Impute missing incidence angles with mean from Train
    inc_mean = np.nanmean(inc_train)
    inc_train = np.nan_to_num(inc_train, nan=inc_mean)
    inc_test = np.nan_to_num(inc_test, nan=inc_mean)

    # Transpose to PyTorch format: (N, C, H, W)
    X_train = X_train.transpose(0, 3, 1, 2)
    X_test = X_test.transpose(0, 3, 1, 2)

    print(f"Saving processed data to {cache_path}")
    np.savez(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train,
        X_test=X_test,
        inc_test=inc_test,
        ids_test=test_ids,
    )

    return X_train, y_train, inc_train, X_test, inc_test, test_ids


class IcebergDataset(Dataset):
    def __init__(self, X, y, inc, transform=False):
        self.X = torch.FloatTensor(X)
        self.inc = torch.FloatTensor(inc)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        inc = self.inc[idx]

        if self.transform:
            # Random Rotation (0, 90, 180, 270 degrees)
            k = np.random.randint(0, 4)
            img = torch.rot90(img, k, [1, 2])

            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                img = torch.flip(img, [2])

        if self.y is not None:
            return img, inc, self.y[idx]
        else:
            return img, inc


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================


class DualStreamPooling(nn.Module):
    """
    Implements Max Pooling and Min Pooling in parallel, concatenating results.
    Min Pooling is implemented as -MaxPool(-x).
    """

    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(DualStreamPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x_max = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        x_min = -F.max_pool2d(-x, self.kernel_size, self.stride, self.padding)
        return torch.cat([x_max, x_min], dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

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
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    def __init__(self, planes, ratio=16):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class HRDSNet(nn.Module):
    """
    High-Resolution Dual-Stream Network.
    Features:
    - Sustained Width Backbone (128 filters)
    - Dual-Stream Pooling (Max+Min)
    - High-Resolution Terminal Block (Stride 1)
    - Split-Path Readout (Strided Conv + Global Avg)
    """

    def __init__(self):
        super(HRDSNet, self).__init__()

        # --- Visual Branch ---

        # Stage 1: 75 -> 37
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.cbam1 = CBAMBlock(128)
        self.pool1 = DualStreamPooling(2, 2, 0)  # Output: 256ch

        # Stage 2: 37 -> 18
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.cbam2 = CBAMBlock(128)
        self.pool2 = DualStreamPooling(2, 2, 0)  # Output: 256ch

        # Stage 3: 18 -> 9
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAMBlock(128)
        self.pool3 = DualStreamPooling(2, 2, 0)  # Output: 256ch

        # Stage 4: 9 -> 9 (Preserve Resolution)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.cbam4 = CBAMBlock(128)
        # Stride 1, Padding 1 to maintain 9x9 size
        self.pool4 = DualStreamPooling(3, 1, 1)  # Output: 256ch

        # Readout Path A: Strided Convolution
        # 9x9 -> 4x4
        self.readout_conv = nn.Conv2d(256, 48, kernel_size=3, stride=2, padding=0)
        # Flattened: 4 * 4 * 48 = 768

        # Readout Path B: Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 256

        # --- Metadata Branch ---
        self.meta_fc1 = nn.Linear(1, 32)
        self.meta_bn1 = nn.BatchNorm1d(32)
        self.meta_fc2 = nn.Linear(32, 32)
        self.meta_bn2 = nn.BatchNorm1d(32)

        # --- Fusion Head ---
        # Visual (768 + 256) + Meta (32) = 1056
        self.head_fc = nn.Linear(1056, 256)
        self.head_bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.out = nn.Linear(256, 1)

    def forward(self, img, inc_angle):
        # Stage 1
        x = F.relu(self.bn1(self.conv1(img)))
        x = self.cbam1(x)
        x = self.pool1(x)

        # Stage 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = self.pool2(x)

        # Stage 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        x = self.pool3(x)

        # Stage 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)
        x = self.pool4(x)

        # Readout Path A
        path_a = self.readout_conv(x)
        path_a = path_a.view(path_a.size(0), -1)

        # Readout Path B
        path_b = self.global_pool(x).view(x.size(0), -1)

        # Visual Fusion
        visual_feat = torch.cat([path_a, path_b], dim=1)

        # Metadata
        m = inc_angle.view(-1, 1)
        m = F.relu(self.meta_bn1(self.meta_fc1(m)))
        m = F.relu(self.meta_bn2(self.meta_fc2(m)))

        # Global Fusion
        fused = torch.cat([visual_feat, m], dim=1)

        # Classification Head
        h = F.relu(self.head_bn(self.head_fc(fused)))
        h = self.dropout(h)
        out = self.out(h)

        return out


# ==========================================
# 3. TRAINING & INFERENCE
# ==========================================


def train_model():
    """
    Executes the full training pipeline:
    1. Loads and processes data.
    2. Runs Stratified 5-Fold Cross-Validation.
    3. Trains HRDS-Net models.
    4. Generates and saves submission file.
    """
    set_seed(SEED)

    # Load Data
    X_train, y_train, inc_train, X_test, inc_test, test_ids = load_and_process_data()

    # Stratified 5-Fold CV
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # Array to store accumulated test predictions
    test_preds_accum = np.zeros(len(test_ids))

    print(f"\nStarting training on {DEVICE}...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n=== Fold {fold} ===")

        # Prepare Fold Data
        X_tr, y_tr, inc_tr = (
            X_train[train_idx],
            y_train[train_idx],
            inc_train[train_idx],
        )
        X_val, y_val, inc_val = X_train[val_idx], y_train[val_idx], inc_train[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(X_tr, y_tr, inc_tr, transform=True)
        val_ds = IcebergDataset(X_val, y_val, inc_val, transform=False)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        # Initialize Model
        model = HRDSNet().to(DEVICE)

        # Optimizer (Adam) and Scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Loss Function
        criterion = nn.BCEWithLogitsLoss()

        # Early Stopping
        checkpoint_path = os.path.join(WORKING_DIR, f"model_fold_{fold}.pth")
        early_stopping = EarlyStopping(
            patience=PATIENCE, verbose=True, path=checkpoint_path
        )

        # Training Loop
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0

            for imgs, incs, labels in train_loader:
                imgs = imgs.to(DEVICE)
                incs = incs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(imgs, incs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * imgs.size(0)

            train_loss /= len(train_ds)

            # Validation Loop
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for imgs, incs, labels in val_loader:
                    imgs = imgs.to(DEVICE)
                    incs = incs.to(DEVICE)
                    labels = labels.to(DEVICE).unsqueeze(1)

                    outputs = model(imgs, incs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)

                    preds = torch.sigmoid(outputs) > 0.5
                    correct += (preds == (labels > 0.5)).sum().item()
                    total += labels.size(0)

            val_loss /= len(val_ds)
            val_acc = correct / total

            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val Acc: {val_acc:.6f}"
            )

            # Step Scheduler & Early Stopping
            scheduler.step(val_loss)
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # Load Best Model for Inference
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        # Predict on Test Set for this Fold
        test_ds = IcebergDataset(X_test, None, inc_test, transform=False)
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        fold_preds = []
        with torch.no_grad():
            for imgs, incs in test_loader:
                imgs = imgs.to(DEVICE)
                incs = incs.to(DEVICE)
                outputs = model(imgs, incs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_preds.extend(probs)

        test_preds_accum += np.array(fold_preds)

    # Average Predictions across Folds
    final_preds = test_preds_accum / NUM_FOLDS

    # Save Submission
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": final_preds})
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
