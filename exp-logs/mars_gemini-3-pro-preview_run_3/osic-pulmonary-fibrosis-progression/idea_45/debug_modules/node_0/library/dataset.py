import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.preprocessing import prepare_data


class LungDataset(Dataset):
    """
    PyTorch Dataset for Lung Function Decline Prediction.

    Serves:
        - Image: (3, 260, 260) tensor from cached numpy arrays.
        - Clinical Features: (5,) tensor containing [Baseline_FVC_scaled, Relative_Time, Age_scaled, Sex, Smoking].
        - Target: (1,) tensor containing Z-score normalized FVC (for train/val).
    """

    def __init__(self, df, image_cache, mode="train"):
        self.df = df.reset_index(drop=True)
        self.image_cache = image_cache
        self.mode = mode
        # Explicit mapping for Sex (not handled in preprocessing)
        self.sex_map = {"Male": 0, "Female": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]

        # 1. Image Data
        # Retrieve from cache. If missing (edge case), return zero tensor to prevent crash.
        if pid in self.image_cache:
            img_data = self.image_cache[pid]
        else:
            img_data = np.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        # Convert to float tensor
        img_tensor = torch.from_numpy(img_data).float()

        # 2. Clinical Features
        # Vector: [Baseline_FVC_scaled, Relative_Time, Age_scaled, Sex_Code, SmokingStatus_Code]

        # Baseline FVC (Z-scored in preprocessing)
        baseline_fvc = float(row["Baseline_FVC_scaled"])

        # Relative Time (Scaled by 0.01 in preprocessing)
        rel_time = float(row["Relative_Time"])

        # Age (Z-scored in preprocessing)
        age = float(row["Age_scaled"])

        # Sex (Manual Encoding)
        sex_raw = row["Sex"]
        sex = float(self.sex_map.get(sex_raw, 0.0))  # Default to 0 if unknown

        # Smoking Status (Encoded in preprocessing: 0=Never, 1=Ex, 2=Current)
        smoking = float(row["SmokingStatus_Code"])

        clinical_features = torch.tensor(
            [baseline_fvc, rel_time, age, sex, smoking], dtype=torch.float32
        )

        # 3. Target
        if self.mode in ["train", "val"]:
            # FVC_scaled is Z-scored using global training stats in preprocessing
            target = torch.tensor([float(row["FVC_scaled"])], dtype=torch.float32)
            return img_tensor, clinical_features, target
        else:
            # Test mode: Return only inputs
            return img_tensor, clinical_features


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=4, load_cached_data=True):
    """
    Prepares DataLoaders for training, validation, and testing.

    Implements caching logic:
    1. Calls `prepare_data` which handles image caching (.npy) and tabular processing.
    2. Explicitly saves processed DataFrames and Target Stats to disk (Parquet/JSON)
       to satisfy artifact requirements and allow inspection.
    """

    # Define cache paths for tabular artifacts
    cache_dir = Config.CACHE_DIR
    train_path = os.path.join(cache_dir, "train_processed.parquet")
    val_path = os.path.join(cache_dir, "val_processed.parquet")
    test_path = os.path.join(cache_dir, "test_processed.parquet")
    stats_path = os.path.join(cache_dir, "target_stats.json")

    # Execute Data Preparation
    # This function loads metadata, processes images (saving to .npy cache),
    # and performs tabular engineering (Target Normalization, Feature Scaling).
    train_df, val_df, test_df, image_cache, target_stats = prepare_data(
        load_cached=load_cached_data
    )

    # Save processed tabular data and stats to cache directory
    # This ensures we have a record of the exact data used for training
    os.makedirs(cache_dir, exist_ok=True)
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    test_df.to_parquet(test_path)
    with open(stats_path, "w") as f:
        json.dump(target_stats, f)

    # Instantiate Datasets
    train_ds = LungDataset(train_df, image_cache, mode="train")
    val_ds = LungDataset(val_df, image_cache, mode="val")
    test_ds = LungDataset(test_df, image_cache, mode="test")

    # Instantiate DataLoaders
    # Pin memory enabled for faster host-to-device transfer
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to maintain batch statistics
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, target_stats
