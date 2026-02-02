import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.utils import INPUT_DIR, METADATA_DIR, WORKING_DIR

# Ensure working directory exists for caching
os.makedirs(WORKING_DIR, exist_ok=True)


def _load_data_with_caching(metadata_path, cache_name, load_cached_data=True):
    """
    Loads image data and labels/ids, using caching to speed up subsequent runs.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_name (str): Unique identifier for the cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images_array, targets_array)
            - images_array: (N, 32, 32, 3) uint8 array
            - targets_array: (N,) array (float32 for labels, object/string for IDs)
    """
    images_cache_path = os.path.join(WORKING_DIR, f"{cache_name}_images.npy")
    targets_cache_path = os.path.join(WORKING_DIR, f"{cache_name}_targets.npy")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(images_cache_path)
        and os.path.exists(targets_cache_path)
    ):
        try:
            images = np.load(images_cache_path)
            targets = np.load(targets_cache_path, allow_pickle=True)
            return images, targets
        except Exception:
            pass  # Fallback to loading from source if cache is corrupt

    # 2. Load from source
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    img_list = []
    target_list = []

    # Pre-allocate if possible or just append (dataset is small ~14k, append is fine)
    # Using list comprehension for speed where possible, but we need error handling for images

    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        img = cv2.imread(full_path)
        if img is None:
            # Fallback for missing images: black image
            img = np.zeros((32, 32, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_list.append(img)

        # If test, we want the ID. If train/val, we want the label.
        # We determine this by checking if it's the test set based on cache_name or column presence
        # However, the caller usually knows. We'll store what's in 'has_cactus' or 'id'.
        # For consistency, let's store 'id' for test and 'has_cactus' for train/val.

        if "test" in cache_name:
            target_list.append(row["id"])
        else:
            target_list.append(row["has_cactus"])

    images = np.array(img_list, dtype=np.uint8)
    targets = np.array(target_list)  # float for labels, string for IDs

    # 3. Save to cache
    np.save(images_cache_path, images)
    np.save(targets_cache_path, targets)

    return images, targets


class CactusDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        cache_name,
        is_train=False,
        is_test=False,
        load_cached_data=True,
    ):
        """
        Args:
            metadata_path (str): Path to metadata CSV.
            cache_name (str): Prefix for cache files.
            is_train (bool): If True, applies augmentation.
            is_test (bool): If True, returns (image, id). Else returns (image, label).
            load_cached_data (bool): Whether to use cached .npy files.
        """
        self.is_train = is_train
        self.is_test = is_test

        self.images, self.targets = _load_data_with_caching(
            metadata_path, cache_name, load_cached_data
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image
        image = self.images[idx]  # (32, 32, 3) uint8

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Augmentation (Only for training)
        if self.is_train:
            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                image = cv2.flip(image, 1)
            # Random Vertical Flip
            if np.random.rand() > 0.5:
                image = cv2.flip(image, 0)

        # To Tensor: (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))

        if self.is_test:
            # Return image and ID
            return image_tensor, self.targets[idx]
        else:
            # Return image and label
            label = torch.tensor(self.targets[idx], dtype=torch.float32)
            return image_tensor, label


def get_dataloaders(batch_size, num_workers=2, load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_meta = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Initialize Datasets
    train_dataset = CactusDataset(
        train_meta,
        cache_name="train",
        is_train=True,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    val_dataset = CactusDataset(
        val_meta,
        cache_name="val",
        is_train=False,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    test_dataset = CactusDataset(
        test_meta,
        cache_name="test",
        is_train=False,
        is_test=True,
        load_cached_data=load_cached_data,
    )

    # Initialize Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # For test, we use batch_size=1 or larger.
    # Since the main loop might expect iteration over images for TTA,
    # we provide a loader. The main loop in utils.py seems to iterate manually
    # or use a loader. Standard practice is a loader.
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Often 1 for TTA pipelines unless custom collate used
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
