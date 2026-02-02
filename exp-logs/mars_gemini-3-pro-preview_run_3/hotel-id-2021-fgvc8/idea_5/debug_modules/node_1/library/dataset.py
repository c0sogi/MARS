import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import seed_everything


class HotelDataset(Dataset):
    """
    PyTorch Dataset for Hotel Identification.
    Handles loading images from disk and applying transformations.
    """

    def __init__(self, df, transform=None, is_test=False, data_root=Config.INPUT_DIR):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_path, label_idx).
            transform (A.Compose): Albumentations transforms.
            is_test (bool): Whether this is the test set (no labels).
            data_root (str): Root directory for images.
        """
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.data_root = data_root

        # Pre-extract paths for faster access
        self.file_paths = df["file_path"].values

        if not self.is_test:
            # Ensure label_idx exists
            if "label_idx" not in df.columns:
                raise ValueError(
                    "DataFrame must contain 'label_idx' column for training/validation."
                )
            self.labels = df["label_idx"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        full_path = os.path.join(self.data_root, file_path)

        # Load image using OpenCV
        image = cv2.imread(full_path)

        # Robustness check: if image fails to load, create a black image
        # (This handles rare corrupted files if any, though metadata check passed)
        if image is None:
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.is_test:
            # Return dummy label for test set
            return image, torch.tensor(0, dtype=torch.long)
        else:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)


def get_transforms(mode="train", image_size=Config.IMAGE_SIZE):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
        image_size (int): Target image resolution.
    """
    if mode == "train":
        # Mild Augmentation Strategy:
        # Resize slightly larger -> Random Crop -> Horizontal Flip
        resize_dim = int(image_size * 1.1)
        return A.Compose(
            [
                A.Resize(height=resize_dim, width=resize_dim),
                A.RandomCrop(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Deterministic preprocessing for Val/Test
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    train_csv_path=Config.TRAIN_CSV,
    val_csv_path=Config.VAL_CSV,
    test_csv_path=Config.TEST_CSV,
    batch_size=None,
    num_workers=None,
    load_cached_data=True,
    debug=None,
    debug_sample_size=None,
):
    """
    Prepares and returns DataLoaders for Train, Validation, and Test sets.

    Features:
    - Caches label encoding to ensure consistency.
    - Supports debug mode for quick iteration.
    - Handles label mapping.

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS
    if debug is None:
        debug = Config.DEBUG
    if debug_sample_size is None:
        debug_sample_size = Config.DEBUG_SAMPLE_SIZE

    seed_everything(Config.SEED)

    # ---------------------------------------------------------
    # 1. Load Metadata
    # ---------------------------------------------------------
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # ---------------------------------------------------------
    # 2. Label Encoding (with Caching)
    # ---------------------------------------------------------
    # We map hotel_id (arbitrary integers) to contiguous class indices (0..N-1).
    # We must use the full training set to define this mapping to ensure
    # all classes are covered, even if we are in debug mode later.

    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    encoder_path = os.path.join(cache_dir, "label_encoder.parquet")

    if load_cached_data and os.path.exists(encoder_path):
        label_map_df = pd.read_parquet(encoder_path)
    else:
        # Generate mapping from full training data
        unique_hotels = sorted(train_df["hotel_id"].unique())
        label_map_df = pd.DataFrame(
            {"hotel_id": unique_hotels, "label_idx": range(len(unique_hotels))}
        )
        # Save to cache
        label_map_df.to_parquet(encoder_path, index=False)

    # Create dictionary for fast mapping
    hotel_to_idx = dict(zip(label_map_df["hotel_id"], label_map_df["label_idx"]))
    num_classes = len(hotel_to_idx)

    # ---------------------------------------------------------
    # 3. Apply Mapping
    # ---------------------------------------------------------
    # Map hotel_id to label_idx for Train and Val
    train_df["label_idx"] = train_df["hotel_id"].map(hotel_to_idx)
    val_df["label_idx"] = val_df["hotel_id"].map(hotel_to_idx)

    # Verify integrity
    if train_df["label_idx"].isnull().any():
        raise ValueError(
            "Found hotel_ids in training set that are not in the label map!"
        )
    if val_df["label_idx"].isnull().any():
        # This theoretically shouldn't happen if Val is subset of Train classes
        raise ValueError(
            "Found hotel_ids in validation set that are not in the label map!"
        )

    # ---------------------------------------------------------
    # 4. Debug Sampling
    # ---------------------------------------------------------
    if debug:
        train_df = train_df.sample(
            n=min(len(train_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 5. Create Datasets
    # ---------------------------------------------------------
    train_dataset = HotelDataset(
        train_df,
        transform=get_transforms(mode="train", image_size=Config.IMAGE_SIZE),
        is_test=False,
    )

    val_dataset = HotelDataset(
        val_df,
        transform=get_transforms(mode="val", image_size=Config.IMAGE_SIZE),
        is_test=False,
    )

    test_dataset = HotelDataset(
        test_df,
        transform=get_transforms(mode="test", image_size=Config.IMAGE_SIZE),
        is_test=True,
    )

    # ---------------------------------------------------------
    # 6. Create DataLoaders
    # ---------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,  # Important for BN stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, num_classes
