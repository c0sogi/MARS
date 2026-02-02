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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold

from library.layers import WideConvBlock, CBAM, DualPooling
from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    MODEL_DIR,
    SUBMISSION_FILE,
    SEED,
    DEVICE,
    NUM_FOLDS,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    NUM_WORKERS,
)
from library.utils import set_seed, calculate_global_stats

# ==========================================
# MODEL DEFINITION
# ==========================================


class MetadataBranch(nn.Module):
    def __init__(self, hidden_dim=32, out_dim=32):
        super(MetadataBranch, self).__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: (batch_size, ) or (batch_size, 1)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn(x)
        x = self.relu2(x)
        return x


class RDP_WBN(nn.Module):
    """
    Robust Dual-Path Wide-Body Network.
    Cite {solution_lesson_node_00123}: Dual-Path Readout (Spatial + GAP).
    Cite {solution_lesson_node_00162}: No pre-fusion BN on branches.
    """

    def __init__(self):
        super(RDP_WBN, self).__init__()

        # --- Visual Branch ---
        # Stage 1: Input 3ch -> Conv(128) -> CBAM -> DualPool(256)
        # Note: DualPooling doubles channels.
        self.stage1_conv = WideConvBlock(3, 128)
        self.stage1_cbam = CBAM(128)
        self.stage1_pool = DualPooling(kernel_size=2, stride=2)

        # Stage 2: Input 256ch -> Conv(128) -> CBAM -> DualPool(256)
        # Delayed Integration: 256 -> 128
        self.stage2_conv = WideConvBlock(256, 128)
        self.stage2_cbam = CBAM(128)
        self.stage2_pool = DualPooling(kernel_size=2, stride=2)

        # Stage 3: Input 256ch -> Conv(128) -> CBAM -> DualPool(256)
        self.stage3_conv = WideConvBlock(256, 128)
        self.stage3_cbam = CBAM(128)
        self.stage3_pool = DualPooling(kernel_size=2, stride=2)

        # Stage 4: Input 256ch -> Conv(128) -> CBAM -> DualPool(256)
        self.stage4_conv = WideConvBlock(256, 128)
        self.stage4_cbam = CBAM(128)
        self.stage4_pool = DualPooling(kernel_size=2, stride=2)

        # --- Dual-Path Readout ---
        # Stage 4 output is 256 channels (128 Peak, 128 Shadow), 4x4 spatial

        # Path A: Spatial Context
        # Conv 256 -> 64, 3x3, pad 1. Flatten.
        # Cite {solution_lesson_node_00162}: No BN here.
        self.path_a_conv = nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False)
        # Output: 64 * 4 * 4 = 1024

        # Path B: Global Intensity
        # GAP 256.
        # Cite {solution_lesson_node_00162}: No BN here.

        # --- Metadata Branch ---
        # Cite {solution_lesson_node_00158}: MLP with hidden layer.
        self.meta_branch = MetadataBranch(hidden_dim=32, out_dim=32)

        # --- Fusion Head ---
        # Input: 1024 (A) + 256 (B) + 32 (Meta) = 1312
        self.fusion_fc = nn.Linear(1312, 512)
        self.fusion_bn = nn.BatchNorm1d(512)
        self.fusion_relu = nn.ReLU(inplace=True)
        self.fusion_dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x_img, x_meta):
        # --- Visual Forward ---
        # Stage 1
        x = self.stage1_conv(x_img)
        x = self.stage1_cbam(x)
        x = self.stage1_pool(x)

        # Stage 2
        x = self.stage2_conv(x)
        x = self.stage2_cbam(x)
        x = self.stage2_pool(x)

        # Stage 3
        x = self.stage3_conv(x)
        x = self.stage3_cbam(x)
        x = self.stage3_pool(x)

        # Stage 4
        x = self.stage4_conv(x)
        x = self.stage4_cbam(x)
        x = self.stage4_pool(x)  # (B, 256, 4, 4)

        # --- Readout ---

        # Path A: Spatial
        feat_a = self.path_a_conv(x)  # (B, 64, 4, 4)
        feat_a = feat_a.view(feat_a.size(0), -1)  # (B, 1024)

        # Path B: Global Intensity
        feat_b = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # (B, 256)

        # --- Metadata Forward ---
        feat_meta = self.meta_branch(x_meta)  # (B, 32)

        # --- Fusion ---
        # Cite {solution_lesson_node_00162}: Concatenate raw features.
        fused = torch.cat([feat_a, feat_b, feat_meta], dim=1)  # (B, 1312)

        out = self.fusion_fc(fused)
        out = self.fusion_bn(out)
        out = self.fusion_relu(out)
        out = self.fusion_dropout(out)
        logits = self.classifier(out)

        return torch.sigmoid(logits)


# ==========================================
# DATASET
# ==========================================


class IcebergDataset(Dataset):
    def __init__(self, data_list, global_stats, augment=False, inc_angle_mean=39.26):
        """
        Args:
            data_list: List of dictionaries (from json).
            global_stats: Dict with min/max for normalization.
            augment: Boolean, enable augmentation.
            inc_angle_mean: Float, value to impute missing inc_angles.
        """
        self.data = data_list
        self.stats = global_stats
        self.augment = augment
        self.inc_angle_mean = inc_angle_mean

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. Image Processing
        # Flattened lists to 75x75 arrays
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
        b3 = (b1 + b2) / 2.0

        # Global Normalization (Min-Max)
        # (x - min) / (max - min)
        b1 = (b1 - self.stats["b1_min"]) / (self.stats["b1_max"] - self.stats["b1_min"])
        b2 = (b2 - self.stats["b2_min"]) / (self.stats["b2_max"] - self.stats["b2_min"])
        b3 = (b3 - self.stats["b3_min"]) / (self.stats["b3_max"] - self.stats["b3_min"])

        # Stack to (3, 75, 75)
        img = np.stack([b1, b2, b3], axis=0)
        img_tensor = torch.from_numpy(img)

        # 2. Augmentation
        if self.augment:
            # Random Rotation (0, 90, 180, 270)
            k = random.choice([0, 1, 2, 3])
            img_tensor = torch.rot90(img_tensor, k, [1, 2])

            # Horizontal Flip
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])

        # 3. Metadata Processing
        inc_angle = item["inc_angle"]
        if inc_angle == "na" or inc_angle is None or pd.isna(inc_angle):
            inc_angle = self.inc_angle_mean
        else:
            inc_angle = float(inc_angle)

        inc_angle_tensor = torch.tensor([inc_angle], dtype=torch.float32)

        # 4. Target
        if "is_iceberg" in item:
            target = torch.tensor([item["is_iceberg"]], dtype=torch.float32)
            return img_tensor, inc_angle_tensor, target, item["id"]
        else:
            return img_tensor, inc_angle_tensor, item["id"]


# ==========================================
# TRAINING & INFERENCE UTILS
# ==========================================


def load_data(debug=False):
    # Load raw JSON
    with open(TRAIN_JSON, "r") as f:
        train_data_raw = json.load(f)

    with open(TEST_JSON, "r") as f:
        test_data_raw = json.load(f)

    # Load metadata for IDs (though we use full train_data_raw for CV)
    # We just need to ensure we have the data.

    # Calculate stats
    stats = calculate_global_stats(load_cached_data=True)

    # Calculate inc_angle mean from training data
    angles = []
    for x in train_data_raw:
        if x["inc_angle"] != "na":
            angles.append(float(x["inc_angle"]))
    inc_mean = np.mean(angles) if angles else 39.26

    if debug:
        train_data_raw = train_data_raw[:100]
        test_data_raw = test_data_raw[:50]

    return train_data_raw, test_data_raw, stats, inc_mean


def train_model(debug=False):
    set_seed(SEED)

    # Load Data
    full_train_data, _, stats, inc_mean = load_data(debug)

    # Prepare Labels for Stratified Split
    labels = [x["is_iceberg"] for x in full_train_data]
    ids = [x["id"] for x in full_train_data]

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # Training Loop
    fold_perf = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(ids, labels)):
        print(f"\n=== Fold {fold} ===")

        # Split Data
        train_subset = [full_train_data[i] for i in train_idx]
        val_subset = [full_train_data[i] for i in val_idx]

        # Datasets & Loaders
        train_ds = IcebergDataset(
            train_subset, stats, augment=True, inc_angle_mean=inc_mean
        )
        val_ds = IcebergDataset(
            val_subset, stats, augment=False, inc_angle_mean=inc_mean
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Model Setup
        model = RDP_WBN().to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        criterion = nn.BCELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        # Epoch Loop
        for epoch in range(MAX_EPOCHS):
            # Train
            model.train()
            train_loss = 0.0
            for imgs, metas, targets, _ in train_loader:
                imgs, metas, targets = (
                    imgs.to(DEVICE),
                    metas.to(DEVICE),
                    targets.to(DEVICE),
                )

                optimizer.zero_grad()
                outputs = model(imgs, metas)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * imgs.size(0)

            train_loss /= len(train_ds)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, metas, targets, _ in val_loader:
                    imgs, metas, targets = (
                        imgs.to(DEVICE),
                        metas.to(DEVICE),
                        targets.to(DEVICE),
                    )
                    outputs = model(imgs, metas)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * imgs.size(0)

            val_loss /= len(val_ds)

            # Scheduler
            scheduler.step(val_loss)

            # Logging
            print(
                f"Epoch {epoch+1}/{MAX_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Save Best Model for Fold
        save_path = os.path.join(MODEL_DIR, f"model_fold_{fold}.pth")
        torch.save(best_model_state, save_path)
        print(f"Saved best model for fold {fold} with Val Loss: {best_val_loss:.6f}")
        fold_perf.append(best_val_loss)

    print(f"\nAverage CV Loss: {np.mean(fold_perf):.6f}")


def predict_and_submit(debug=False):
    set_seed(SEED)

    # Load Data
    _, test_data_raw, stats, inc_mean = load_data(debug)
    test_ds = IcebergDataset(
        test_data_raw, stats, augment=False, inc_angle_mean=inc_mean
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Load Models
    models = []
    for fold in range(NUM_FOLDS):
        path = os.path.join(MODEL_DIR, f"model_fold_{fold}.pth")
        if os.path.exists(path):
            model = RDP_WBN().to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            models.append(model)
        else:
            print(f"Warning: Model for fold {fold} not found at {path}")

    if not models:
        print("No models found. Cannot generate submission.")
        return

    # Inference
    results = []
    print("Starting Inference...")
    with torch.no_grad():
        for imgs, metas, ids in test_loader:
            imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)

            batch_preds = []
            for model in models:
                preds = model(imgs, metas)
                batch_preds.append(preds.cpu().numpy())

            # Average predictions across folds
            avg_preds = np.mean(batch_preds, axis=0)

            for i, pred in enumerate(avg_preds):
                results.append({"id": ids[i], "is_iceberg": float(pred[0])})

    # Save Submission
    df_sub = pd.DataFrame(results)
    df_sub.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")
