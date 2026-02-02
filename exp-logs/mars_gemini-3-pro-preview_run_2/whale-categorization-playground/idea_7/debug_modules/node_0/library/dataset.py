import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# -------------------------------------------------------------------------
# Constants & Configuration
# -------------------------------------------------------------------------
# Dataset specific statistics from EDA
MEAN = [0.5479, 0.5816, 0.6211]
STD = [0.2822, 0.2668, 0.2625]


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def get_label_mapping():
    """
    Generates a dictionary mapping Whale IDs to integers.
    Strictly excludes 'new_whale' to match N_CLASSES=4028.
    """
    df_train = pd.read_csv(Config.TRAIN_CSV)

    # Filter out new_whale
    known_whales = df_train[df_train["Id"] != "new_whale"]["Id"].unique()
    known_whales = sorted(known_whales)

    id2idx = {label: i for i, label in enumerate(known_whales)}
    idx2id = {i: label for i, label in enumerate(known_whales)}

    return id2idx, idx2id


def load_cached_images(df, resolution, cache_name, load_cached_data=True):
    """
    Loads images from a numpy cache or processes them from disk.

    Args:
        df: DataFrame containing 'file_path'.
        resolution: Target image size (resolution x resolution).
        cache_name: Unique identifier for the cache file.
        load_cached_data: Boolean to enable/disable loading from cache.

    Returns:
        numpy.ndarray: Array of shape (N, resolution, resolution, 3)
    """
    cache_path = Config.get_cache_path(f"{cache_name}_{resolution}")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached images from {cache_path}...")
        try:
            images = np.load(cache_path)
            if images.shape[0] == len(df):
                return images
            else:
                print(
                    f"Cache mismatch (Size {images.shape[0]} vs {len(df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Process from scratch
    print(
        f"Processing {len(df)} images for {cache_name} at {resolution}x{resolution}..."
    )

    # Pre-allocate array for memory efficiency
    images = np.zeros((len(df), resolution, resolution, 3), dtype=np.uint8)

    for i, row in df.iterrows():
        # Construct full path
        # Note: row['file_path'] is relative to input dir (e.g. "train/xxxx.jpg")
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        img = cv2.imread(full_path)

        if img is None:
            # Fallback for corrupt images (should not happen based on EDA)
            # Use black image
            img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (resolution, resolution))

        images[i] = img

    # 3. Save to cache
    print(f"Saving cache to {cache_path}...")
    np.save(cache_path, images)

    return images


def get_transforms(mode, resolution):
    """
    Returns Albumentations transforms.
    Note: Resizing is handled during data loading/caching.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # Add slight affine to improve robustness
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )
    else:
        # Val / Test
        return A.Compose(
            [
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------
class WhaleDataset(Dataset):
    def __init__(self, images, targets=None, transform=None):
        """
        Args:
            images (np.ndarray): Image data (N, H, W, 3).
            targets (np.ndarray, optional): Integer labels.
            transform: Albumentations transform.
        """
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.targets is not None:
            label = self.targets[idx]
            return image, label
        else:
            return image


# -------------------------------------------------------------------------
# Data Loader Generators
# -------------------------------------------------------------------------
def get_train_val_loaders(resolution, batch_size, load_cached_data=True):
    """
    Prepares DataLoaders for Training and Validation.

    Strategy:
    - Train: Removes 'new_whale'. Only known classes.
    - Val: Keeps 'new_whale' (labeled as -1).
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    # 2. Filter Training Data (Known Only)
    df_train_filtered = df_train[df_train["Id"] != "new_whale"].reset_index(drop=True)

    # 3. Load/Process Images
    train_images = load_cached_images(
        df_train_filtered, resolution, "train_images", load_cached_data
    )

    val_images = load_cached_images(df_val, resolution, "val_images", load_cached_data)

    # 4. Prepare Labels
    id2idx, _ = get_label_mapping()

    # Train labels (All should be in id2idx)
    train_targets = df_train_filtered["Id"].map(id2idx).values.astype(np.int64)

    # Val labels (Handle new_whale as -1)
    val_targets = df_val["Id"].map(lambda x: id2idx.get(x, -1)).values.astype(np.int64)

    # 5. Create Datasets
    train_dataset = WhaleDataset(
        images=train_images,
        targets=train_targets,
        transform=get_transforms("train", resolution),
    )

    val_dataset = WhaleDataset(
        images=val_images,
        targets=val_targets,
        transform=get_transforms("val", resolution),
    )

    # 6. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Important for BatchNorm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(resolution, batch_size, load_cached_data=True):
    """
    Prepares DataLoader for Inference (Test Set).
    """
    # 1. Load Metadata
    df_test = pd.read_csv(Config.TEST_CSV)

    # 2. Load/Process Images
    test_images = load_cached_images(
        df_test, resolution, "test_images", load_cached_data
    )

    # 3. Create Dataset
    test_dataset = WhaleDataset(
        images=test_images,
        targets=None,  # No targets for test
        transform=get_transforms("test", resolution),
    )

    # 4. Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
