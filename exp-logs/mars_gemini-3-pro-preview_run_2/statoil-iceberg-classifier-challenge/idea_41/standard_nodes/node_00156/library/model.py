import os
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import calculate_global_stats, seed_everything
from library.model_components import ConvBnRelu, CBAM, DualPooling


# ==========================================
# DATASET
# ==========================================
class IcebergDataset(Dataset):
    def __init__(
        self,
        metadata_csv,
        json_file,
        transform=None,
        global_stats=None,
        inc_angle_stats=None,
    ):
        """
        Args:
            metadata_csv (str): Path to the metadata CSV file (train.csv, val.csv, or test.csv).
            json_file (str): Path to the corresponding JSON file containing image data.
            transform (bool): Whether to apply augmentation (Rotation/Flip).
            global_stats (dict): Global min/max stats for scaling.
            inc_angle_stats (dict): Mean/Std for incidence angle normalization.
        """
        self.meta = pd.read_csv(metadata_csv)
        self.json_path = json_file
        self.transform = transform
        self.global_stats = global_stats
        self.inc_angle_stats = inc_angle_stats

        # Load JSON data into memory (efficient for this dataset size ~1600 images)
        with open(self.json_path, "r") as f:
            raw_data = json.load(f)

        # Create a lookup dict for fast access by ID
        self.data_map = {item["id"]: item for item in raw_data}

        # Filter metadata to only include IDs present in the JSON (safety check)
        self.meta = self.meta[self.meta["id"].isin(self.data_map.keys())].reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_id = row["id"]
        item = self.data_map[img_id]

        # 1. Load Bands
        band_1 = np.array(item["band_1"]).reshape(75, 75).astype(np.float32)
        band_2 = np.array(item["band_2"]).reshape(75, 75).astype(np.float32)

        # 2. Global Scaling
        # Apply min-max scaling using global stats
        # (x - min) / (max - min)
        b1_min, b1_max = self.global_stats["b1_min"], self.global_stats["b1_max"]
        b2_min, b2_max = self.global_stats["b2_min"], self.global_stats["b2_max"]

        band_1 = (band_1 - b1_min) / (b1_max - b1_min)
        band_2 = (band_2 - b2_min) / (b2_max - b2_min)

        # 3. Construct 3rd Channel (Average)
        # Note: We average the SCALED bands
        band_3 = (band_1 + band_2) / 2.0

        # Stack: (H, W, C) -> (C, H, W) for PyTorch
        img = np.stack([band_1, band_2, band_3], axis=0)

        # 4. Incidence Angle
        # Handle missing values (na) by imputing with mean
        inc_angle = row["inc_angle"]
        if pd.isna(inc_angle) or inc_angle == "na":
            inc_angle = self.inc_angle_stats["mean"]
        else:
            inc_angle = float(inc_angle)

        # Normalize incidence angle
        inc_angle = (inc_angle - self.inc_angle_stats["mean"]) / self.inc_angle_stats[
            "std"
        ]

        # 5. Augmentation (Training Only)
        if self.transform:
            # Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            img = np.rot90(img, k, axes=(1, 2)).copy()

            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=2).copy()

        # Convert to Tensor
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor([inc_angle], dtype=torch.float32)

        # 6. Target
        if "is_iceberg" in row:
            target = torch.tensor([row["is_iceberg"]], dtype=torch.float32)
            return img_tensor, angle_tensor, target
        else:
            return img_tensor, angle_tensor, img_id


# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class NFWBN(nn.Module):
    """
    Normalized-Fusion Wide-Body Network (NF-WBN).
    Features:
    - Wide-Body Backbone (128 filters)
    - Delayed-Integration Blocks
    - CBAM Attention
    - Dual-Stream Pooling (Max+Min)
    - Normalized Dual-Path Readout (Spatial + Intensity)
    - Normalized Metadata Fusion
    """

    def __init__(self):
        super(NFWBN, self).__init__()

        # --- Visual Branch ---
        # Stage 1
        # Input: 3 channels. Output: 128 filters -> Pool -> 256 channels
        self.stage1 = nn.Sequential(
            ConvBnRelu(Config.IN_CHANNELS, Config.MODEL_FILTERS),
            CBAM(Config.MODEL_FILTERS),
            DualPooling(),
        )

        # Stage 2
        # Input: 256 channels (from DualPooling). Map to 128 (Delayed Integration).
        self.stage2 = nn.Sequential(
            ConvBnRelu(Config.MODEL_FILTERS * 2, Config.MODEL_FILTERS),
            CBAM(Config.MODEL_FILTERS),
            DualPooling(),
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBnRelu(Config.MODEL_FILTERS * 2, Config.MODEL_FILTERS),
            CBAM(Config.MODEL_FILTERS),
            DualPooling(),
        )

        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBnRelu(Config.MODEL_FILTERS * 2, Config.MODEL_FILTERS),
            CBAM(Config.MODEL_FILTERS),
            DualPooling(),
        )
        # Output of Stage 4 is (Batch, 256, 4, 4) given 75x75 input

        # --- Normalized Dual-Path Readout ---
        # Path A: Spatial Context
        # Map 256 -> 48 channels, preserve spatial 4x4. Flatten -> 48*4*4 = 768
        self.path_a_conv = nn.Conv2d(256, 48, kernel_size=3, padding=1)
        self.path_a_bn = nn.BatchNorm1d(48 * 4 * 4)

        # Path B: Robust Intensity
        # Global Average Pooling 256 -> 256
        self.path_b_gap = nn.AdaptiveAvgPool2d(1)
        self.path_b_bn = nn.BatchNorm1d(256)

        # --- Metadata Branch ---
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, Config.META_EMBED_DIM),
            nn.BatchNorm1d(Config.META_EMBED_DIM),
            nn.ReLU(inplace=True),
        )

        # --- Fusion Head ---
        # Input: 768 (Spatial) + 256 (Intensity) + 32 (Meta) = 1056
        fusion_dim = (48 * 4 * 4) + 256 + Config.META_EMBED_DIM

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 1),
        )

    def forward(self, x_img, x_angle):
        # Visual Backbone
        x = self.stage1(x_img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)  # Shape: (B, 256, 4, 4)

        # Path A: Spatial
        x_spatial = self.path_a_conv(x)
        x_spatial = x_spatial.view(x_spatial.size(0), -1)  # Flatten
        x_spatial = self.path_a_bn(x_spatial)

        # Path B: Intensity
        x_intensity = self.path_b_gap(x)
        x_intensity = x_intensity.view(x_intensity.size(0), -1)
        x_intensity = self.path_b_bn(x_intensity)

        # Metadata
        x_meta = self.meta_mlp(x_angle)

        # Fusion
        x_fused = torch.cat([x_spatial, x_intensity, x_meta], dim=1)
        logits = self.fusion_head(x_fused)

        return logits


# ==========================================
# TRAINING & INFERENCE UTILS
# ==========================================
def get_inc_angle_stats(metadata_path):
    """Computes mean/std of inc_angle from training metadata for normalization."""
    df = pd.read_csv(metadata_path)
    # Filter valid angles
    valid_angles = pd.to_numeric(df["inc_angle"], errors="coerce").dropna()
    return {"mean": valid_angles.mean(), "std": valid_angles.std()}


def train_nfwbn():
    """
    Executes the Stratified 5-Fold Training of NF-WBN.
    """
    seed_everything(Config.SEED)

    # 1. Prepare Stats
    global_stats = calculate_global_stats()
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    inc_angle_stats = get_inc_angle_stats(train_meta_path)

    # 2. Load Full Training Metadata for K-Fold
    # Note: The metadata folder already has 'train.csv' and 'val.csv' which represents
    # a single split. However, the 'Idea' requires 5-Fold Cross Validation.
    # We will combine train and val CSVs to recreate the full dataset, then apply KFold.
    df_train_part = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    df_val_part = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    full_df = pd.concat([df_train_part, df_val_part]).reset_index(drop=True)

    # Stratified K-Fold
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Create temporary metadata files for folds
    fold_dir = os.path.join(Config.WORK_DIR, "folds")
    os.makedirs(fold_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_df, full_df["is_iceberg"])
    ):
        print(f"\n=== Training Fold {fold} ===")

        train_fold_df = full_df.iloc[train_idx]
        val_fold_df = full_df.iloc[val_idx]

        train_csv = os.path.join(fold_dir, f"train_fold_{fold}.csv")
        val_csv = os.path.join(fold_dir, f"val_fold_{fold}.csv")

        train_fold_df.to_csv(train_csv, index=False)
        val_fold_df.to_csv(val_csv, index=False)

        # Datasets
        train_ds = IcebergDataset(
            train_csv,
            os.path.join(Config.INPUT_DIR, "train.json"),
            transform=True,
            global_stats=global_stats,
            inc_angle_stats=inc_angle_stats,
        )
        val_ds = IcebergDataset(
            val_csv,
            os.path.join(Config.INPUT_DIR, "train.json"),
            transform=False,
            global_stats=global_stats,
            inc_angle_stats=inc_angle_stats,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model Setup
        model = NFWBN().to(Config.DEVICE)
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=True,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(Config.NUM_EPOCHS):
            model.train()
            train_loss = 0.0

            for imgs, angles, targets in train_loader:
                imgs, angles, targets = (
                    imgs.to(Config.DEVICE),
                    angles.to(Config.DEVICE),
                    targets.to(Config.DEVICE),
                )

                optimizer.zero_grad()
                outputs = model(imgs, angles)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * imgs.size(0)

            train_loss /= len(train_ds)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, angles, targets in val_loader:
                    imgs, angles, targets = (
                        imgs.to(Config.DEVICE),
                        angles.to(Config.DEVICE),
                        targets.to(Config.DEVICE),
                    )
                    outputs = model(imgs, angles)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * imgs.size(0)

            val_loss /= len(val_ds)

            # Scheduler Step
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            # Early Stopping & Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        # Save best model for this fold
        save_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold}.pth")
        torch.save(best_model_state, save_path)
        print(
            f"Fold {fold} finished. Best Val Loss: {best_val_loss:.6f}. Saved to {save_path}"
        )


def predict_and_submit():
    """
    Generates submission by averaging predictions from all 5 folds.
    """
    seed_everything(Config.SEED)

    # Stats
    global_stats = calculate_global_stats()
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    inc_angle_stats = get_inc_angle_stats(train_meta_path)

    # Dataset
    test_ds = IcebergDataset(
        os.path.join(Config.METADATA_DIR, "test.csv"),
        os.path.join(Config.INPUT_DIR, "test.json"),
        transform=False,
        global_stats=global_stats,
        inc_angle_stats=inc_angle_stats,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Storage for predictions
    # Shape: (Num_Test_Samples, Num_Folds)
    all_preds = np.zeros((len(test_ds), Config.NUM_FOLDS))
    ids = []

    # Collect IDs once
    for _, _, img_id in test_loader:
        ids.extend(img_id)

    # Inference Loop
    for fold in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        print(f"Predicting with model fold {fold}...")
        model = NFWBN().to(Config.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for imgs, angles, _ in test_loader:
                imgs, angles = imgs.to(Config.DEVICE), angles.to(Config.DEVICE)
                logits = model(imgs, angles)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_preds.extend(probs.flatten())

        all_preds[:, fold] = fold_preds

    # Average Predictions
    avg_preds = np.mean(all_preds, axis=1)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": ids, "is_iceberg": avg_preds})

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
