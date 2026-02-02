import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class IcebergDataset(Dataset):
    def __init__(
        self, metadata_df, images_dict, inc_angle_mean, transform=None, mode="train"
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing metadata (id, inc_angle, etc.)
            images_dict (dict): Dictionary mapping 'id' to normalized numpy image arrays (H, W, C).
            inc_angle_mean (float): Mean incidence angle for imputation.
            transform (albumentations.Compose): Augmentation pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.images_dict = images_dict
        self.inc_angle_mean = inc_angle_mean
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_id = row["id"]

        # Retrieve image (H, W, C)
        image = self.images_dict[img_id]

        # Handle incidence angle
        inc_angle = row["inc_angle"]
        if pd.isna(inc_angle) or inc_angle == "na":
            inc_angle = self.inc_angle_mean
        else:
            inc_angle = float(inc_angle)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Default to tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Prepare return dict
        sample = {
            "image": image,  # (C, H, W)
            "inc_angle": torch.tensor([inc_angle], dtype=torch.float32),
            "id": img_id,
        }

        if self.mode != "test":
            label = int(row["is_iceberg"])
            sample["label"] = torch.tensor([label], dtype=torch.float32)

        return sample


def get_transforms(mode="train"):
    """
    Returns albumentations transforms based on the mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),  # Covers 0, 90, 180, 270 degrees
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into 3-channel images, applies global normalization,
    and caches the result.

    Returns:
        images_dict (dict): Map of id -> normalized image array.
        inc_angle_mean (float): Mean incidence angle from training set.
    """
    cache_path = Config.CACHE_FILE

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached data from {cache_path}...")
            data = np.load(cache_path, allow_pickle=True)
            images_dict = data["images_dict"].item()
            inc_angle_mean = float(data["inc_angle_mean"])
            return images_dict, inc_angle_mean
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load raw JSONs
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    # Combine for processing loop, but keep track of source
    # We need training data specifically for stats calculation

    def process_entry(entry):
        # Extract bands
        band_1 = np.array(entry["band_1"]).reshape(75, 75)
        band_2 = np.array(entry["band_2"]).reshape(75, 75)
        # Construct 3rd channel: Mean
        band_3 = (band_1 + band_2) / 2.0
        # Stack: (75, 75, 3)
        img = np.dstack((band_1, band_2, band_3))
        return entry["id"], img, entry.get("inc_angle")

    # Process Train
    train_images = []
    train_ids = []
    train_angles = []

    for entry in train_data:
        eid, img, angle = process_entry(entry)
        train_images.append(img)
        train_ids.append(eid)
        # Collect angles for mean calculation (exclude 'na')
        if angle != "na" and angle is not None:
            train_angles.append(float(angle))

    # Process Test
    test_images = []
    test_ids = []
    for entry in test_data:
        eid, img, _ = process_entry(entry)
        test_images.append(img)
        test_ids.append(eid)

    # Convert to numpy for vectorized stats
    train_images_np = np.array(train_images)  # (N_train, 75, 75, 3)
    test_images_np = np.array(test_images)  # (N_test, 75, 75, 3)

    # Calculate Global Stats from Training Data Only
    # Independent Per-Channel Min-Max
    # Shape: (3,)
    global_min = train_images_np.min(axis=(0, 1, 2))
    global_max = train_images_np.max(axis=(0, 1, 2))

    print(f"Global Min (Train): {global_min}")
    print(f"Global Max (Train): {global_max}")

    # Calculate Inc Angle Mean
    inc_angle_mean = np.mean(train_angles)
    print(f"Incidence Angle Mean: {inc_angle_mean}")

    # Normalize
    # (X - min) / (max - min)
    # Allow values > 1.0 or < 0.0 in test/val if they exceed train bounds (No clipping)
    epsilon = 1e-8  # Prevent div by zero
    denominator = global_max - global_min + epsilon

    train_images_norm = (train_images_np - global_min) / denominator
    test_images_norm = (test_images_np - global_min) / denominator

    # Build Dictionary
    images_dict = {}
    for i, eid in enumerate(train_ids):
        images_dict[eid] = train_images_norm[i].astype(np.float32)
    for i, eid in enumerate(test_ids):
        images_dict[eid] = test_images_norm[i].astype(np.float32)

    # Cache results
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, images_dict=images_dict, inc_angle_mean=inc_angle_mean)
    print(f"Data cached to {cache_path}")

    return images_dict, inc_angle_mean


def make_dataloaders(load_cached_data=True, train_df=None, val_df=None):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    Allows passing custom DataFrames for Cross-Validation.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        train_df (pd.DataFrame): Optional override for training data.
        val_df (pd.DataFrame): Optional override for validation data.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Get Processed Data
    images_dict, inc_angle_mean = process_and_cache_data(load_cached_data)

    # 2. Load Metadata (if not provided)
    if train_df is None:
        train_df = pd.read_csv(Config.TRAIN_META)
    if val_df is None:
        val_df = pd.read_csv(Config.VAL_META)

    df_test = pd.read_csv(Config.TEST_META)

    # Debug mode: subsample
    if Config.DEBUG:
        limit = Config.MAX_SAMPLES if Config.MAX_SAMPLES else 100
        train_df = train_df.head(limit)
        val_df = val_df.head(limit)
        df_test = df_test.head(limit)
        print(f"DEBUG MODE: Subsampled data to {limit} rows.")

    # 3. Create Datasets
    train_dataset = IcebergDataset(
        train_df,
        images_dict,
        inc_angle_mean,
        transform=get_transforms("train"),
        mode="train",
    )

    val_dataset = IcebergDataset(
        val_df, images_dict, inc_angle_mean, transform=get_transforms("val"), mode="val"
    )

    test_dataset = IcebergDataset(
        df_test,
        images_dict,
        inc_angle_mean,
        transform=get_transforms("test"),
        mode="test",
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
