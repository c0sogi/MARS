import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class IcebergDataset(Dataset):
    """
    Custom Dataset for Ship vs Iceberg classification.
    Handles dynamic normalization and tensor conversion.
    """

    def __init__(self, images, angles, labels=None, config=None):
        """
        Args:
            images (np.ndarray): Shape (N, 75, 75, 3)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,). Defaults to None.
            config (Config, optional): Configuration object.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.config = config if config else Config()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Extract data
        img = self.images[idx]  # (75, 75, 3)
        angle = self.angles[idx]

        # Normalization (Min-Max Scaling)
        min_db = self.config.MIN_DB
        max_db = self.config.MAX_DB

        # Apply scaling: (x - min) / (max - min)
        img = (img - min_db) / (max_db - min_db)

        # Convert to Tensor
        # PyTorch expects (C, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            return img_tensor, angle_tensor


def _process_split(metadata_df, raw_data_list, is_test=False, inc_angle_mean=0.0):
    """
    Helper function to process raw JSON data based on metadata indices.
    Generates 3-channel images and handles angle imputation.
    """
    count = len(metadata_df)

    # Pre-allocate arrays for efficiency
    images = np.zeros((count, 75, 75, 3), dtype=np.float32)
    angles = np.zeros((count,), dtype=np.float32)
    labels = np.zeros((count,), dtype=np.float32) if not is_test else None

    # Iterate through metadata to map samples
    for i, row in metadata_df.iterrows():
        # 'sample_index' in metadata points to the index in the raw JSON list
        raw_idx = int(row["sample_index"])
        item = raw_data_list[raw_idx]

        # Process Bands
        # Raw data is flattened list of 5625 floats
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)

        # Construct 3rd Channel (Mean)
        b_avg = (b1 + b2) / 2.0

        # Stack channels
        images[i, :, :, 0] = b1
        images[i, :, :, 1] = b2
        images[i, :, :, 2] = b_avg

        # Process Incidence Angle
        ang = item["inc_angle"]
        if ang == "na" or pd.isna(ang):
            angles[i] = inc_angle_mean
        else:
            try:
                angles[i] = float(ang)
            except (ValueError, TypeError):
                angles[i] = inc_angle_mean

        # Process Label
        if not is_test:
            labels[i] = float(item["is_iceberg"])

    return images, angles, labels


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders.
    Handles caching of processed numpy arrays to disk.
    """
    config = Config()

    # Ensure working directory exists for cache
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = [
        config.CACHE_TRAIN_DATA,
        config.CACHE_TRAIN_ANGLES,
        config.CACHE_TRAIN_LABELS,
        config.CACHE_VAL_DATA,
        config.CACHE_VAL_ANGLES,
        config.CACHE_VAL_LABELS,
        config.CACHE_TEST_DATA,
        config.CACHE_TEST_ANGLES,
        config.CACHE_TEST_IDS,
    ]

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files)

    if load_cached_data and cache_exists:
        # Load from cache
        train_images = np.load(config.CACHE_TRAIN_DATA)
        train_angles = np.load(config.CACHE_TRAIN_ANGLES)
        train_labels = np.load(config.CACHE_TRAIN_LABELS)

        val_images = np.load(config.CACHE_VAL_DATA)
        val_angles = np.load(config.CACHE_VAL_ANGLES)
        val_labels = np.load(config.CACHE_VAL_LABELS)

        test_images = np.load(config.CACHE_TEST_DATA)
        test_angles = np.load(config.CACHE_TEST_ANGLES)
        # test_ids are loaded by the submission script usually, but we ensure they exist

    else:
        # Load Metadata
        df_train = pd.read_csv(config.TRAIN_META_PATH)
        df_val = pd.read_csv(config.VAL_META_PATH)
        df_test = pd.read_csv(config.TEST_META_PATH)

        # Calculate Imputation Value for Incidence Angle
        # We use the mean of the training set (excluding 'na')
        train_angles_numeric = pd.to_numeric(df_train["inc_angle"], errors="coerce")
        inc_angle_mean = train_angles_numeric.mean()

        if pd.isna(inc_angle_mean):
            inc_angle_mean = config.INC_ANGLE_MEAN  # Fallback to config constant

        # Load Raw Train JSON (contains both train and val samples)
        with open(config.TRAIN_JSON, "r") as f:
            train_raw_data = json.load(f)

        # Process Train Split
        train_images, train_angles, train_labels = _process_split(
            df_train, train_raw_data, is_test=False, inc_angle_mean=inc_angle_mean
        )

        # Process Val Split
        val_images, val_angles, val_labels = _process_split(
            df_val, train_raw_data, is_test=False, inc_angle_mean=inc_angle_mean
        )

        # Free memory
        del train_raw_data

        # Load Raw Test JSON
        with open(config.TEST_JSON, "r") as f:
            test_raw_data = json.load(f)

        # Process Test Split
        test_images, test_angles, _ = _process_split(
            df_test, test_raw_data, is_test=True, inc_angle_mean=inc_angle_mean
        )

        # Extract Test IDs for submission mapping
        test_ids = df_test["id"].values

        # Free memory
        del test_raw_data

        # Save processed data to cache
        np.save(config.CACHE_TRAIN_DATA, train_images)
        np.save(config.CACHE_TRAIN_ANGLES, train_angles)
        np.save(config.CACHE_TRAIN_LABELS, train_labels)

        np.save(config.CACHE_VAL_DATA, val_images)
        np.save(config.CACHE_VAL_ANGLES, val_angles)
        np.save(config.CACHE_VAL_LABELS, val_labels)

        np.save(config.CACHE_TEST_DATA, test_images)
        np.save(config.CACHE_TEST_ANGLES, test_angles)
        np.save(config.CACHE_TEST_IDS, test_ids)

    # Create Dataset Objects
    train_dataset = IcebergDataset(train_images, train_angles, train_labels, config)
    val_dataset = IcebergDataset(val_images, val_angles, val_labels, config)
    test_dataset = IcebergDataset(test_images, test_angles, labels=None, config=config)

    # Create DataLoaders
    # num_workers set to 2 for efficient data loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
