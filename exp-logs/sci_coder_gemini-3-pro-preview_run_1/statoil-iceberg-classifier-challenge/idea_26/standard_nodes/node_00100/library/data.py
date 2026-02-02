import os
import json
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("data_module")

# Constants for Angle Normalization (derived from analysis)
ANGLE_MEAN = 39.2829
ANGLE_STD = 3.8362


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=20,
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                A.Resize(
                    height=Config.IMG_SIZE,
                    width=Config.IMG_SIZE,
                    interpolation=cv2.INTER_CUBIC,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(
                    height=Config.IMG_SIZE,
                    width=Config.IMG_SIZE,
                    interpolation=cv2.INTER_CUBIC,
                ),
                ToTensorV2(),
            ]
        )


def process_json(json_path, cache_name, load_cached_data=True):
    """
    Parses the JSON file, processes bands and angles, and caches the result.

    Args:
        json_path (str): Path to the raw .json file.
        cache_name (str): Name of the cache file (e.g., 'train_processed.npz').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary containing processed numpy arrays.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, cache_name)

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached data from {cache_path}")
        try:
            # Allow pickle=True because we are saving/loading object arrays (ids)
            data = np.load(cache_path, allow_pickle=True)
            return {k: data[k] for k in data.files}
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing...")

    logger.info(f"Processing raw data from {json_path}...")
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Initialize lists
    ids = []
    band_1 = []
    band_2 = []
    inc_angles = []
    labels = []
    has_labels = "is_iceberg" in raw_data[0]

    for item in raw_data:
        ids.append(item["id"])

        # Process bands: Flattened list -> 75x75 numpy array
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)
        band_1.append(b1)
        band_2.append(b2)

        # Process angle: Handle 'na' by imputing with global mean
        angle = item["inc_angle"]
        if angle == "na":
            angle = ANGLE_MEAN
        else:
            angle = float(angle)
        inc_angles.append(angle)

        if has_labels:
            labels.append(item["is_iceberg"])

    # Convert to numpy arrays
    ids = np.array(ids)
    band_1 = np.array(band_1, dtype=np.float32)
    band_2 = np.array(band_2, dtype=np.float32)
    inc_angles = np.array(inc_angles, dtype=np.float32)

    result = {"ids": ids, "band_1": band_1, "band_2": band_2, "inc_angles": inc_angles}

    if has_labels:
        result["labels"] = np.array(labels, dtype=np.float32)

    # Save to cache
    logger.info(f"Saving processed data to {cache_path}")
    np.savez(cache_path, **result)

    return result


class IcebergDataset(Dataset):
    def __init__(self, data_dict, indices, transform=None):
        """
        Args:
            data_dict (dict): Dictionary containing full dataset arrays (band_1, band_2, etc.).
            indices (array-like): List/Array of indices to select from the data_dict.
            transform (callable, optional): Albumentations transform pipeline.
        """
        self.indices = indices
        self.band_1 = data_dict["band_1"][indices]
        self.band_2 = data_dict["band_2"][indices]
        self.inc_angles = data_dict["inc_angles"][indices]
        self.ids = data_dict["ids"][indices]

        self.labels = None
        if "labels" in data_dict:
            self.labels = data_dict["labels"][indices]

        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 1. Retrieve Bands
        b1 = self.band_1[idx]
        b2 = self.band_2[idx]

        # 2. Independent Band Normalization [0, 1]
        b1_norm = (b1 - Config.BAND1_MIN) / (Config.BAND1_MAX - Config.BAND1_MIN)
        b2_norm = (b2 - Config.BAND2_MIN) / (Config.BAND2_MAX - Config.BAND2_MIN)

        # 3. Create Composite Band (Average)
        b3_norm = (b1_norm + b2_norm) / 2.0

        # 4. Stack to form (75, 75, 3)
        # Transpose to (H, W, C) for Albumentations
        image = np.dstack((b1_norm, b2_norm, b3_norm)).astype(np.float32)

        # 5. Apply Transforms (Augmentation + Upsampling to 224x224)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # 6. Normalize Angle
        angle = self.inc_angles[idx]
        angle_norm = (angle - ANGLE_MEAN) / ANGLE_STD
        angle_tensor = torch.tensor(angle_norm, dtype=torch.float32)

        # 7. Return
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, angle_tensor, label
        else:
            return image, angle_tensor, self.ids[idx]


def get_dataloaders(
    fold=0,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    num_workers=Config.NUM_WORKERS,
):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        fold (int): Current fold index (not strictly used if using pre-split metadata,
                    but kept for compatibility).
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached .npz files.
        num_workers (int): Number of worker threads.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # 1. Load Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)

    # 2. Process Raw JSONs (Full Data)
    # Train and Val come from train.json
    train_full_data = process_json(
        Config.TRAIN_JSON, "train_processed.npz", load_cached_data=load_cached_data
    )

    # Test comes from test.json
    test_full_data = process_json(
        Config.TEST_JSON, "test_processed.npz", load_cached_data=load_cached_data
    )

    # 3. Extract Indices from Metadata
    # The metadata contains 'sample_index' which maps to the index in the raw json list.
    # Since process_json preserves order, these indices are valid for the arrays.
    train_indices = df_train_meta["sample_index"].values
    val_indices = df_val_meta["sample_index"].values
    test_indices = df_test_meta["sample_index"].values

    # 4. Create Datasets
    train_dataset = IcebergDataset(
        train_full_data, train_indices, transform=get_transforms(mode="train")
    )

    val_dataset = IcebergDataset(
        train_full_data, val_indices, transform=get_transforms(mode="val")
    )

    test_dataset = IcebergDataset(
        test_full_data, test_indices, transform=get_transforms(mode="test")
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
