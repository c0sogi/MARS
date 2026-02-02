import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
)


class CactusDataset(Dataset):
    """
    PyTorch Dataset for the Cactus Identification task.
    """

    def __init__(self, images, targets, ids, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            targets (np.ndarray): Array of targets with shape (N,).
            ids (np.ndarray): Array of image IDs (filenames).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.targets = targets
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image and target
        # Image is (H, W, C) in uint8 [0, 255]
        image = self.images[idx]
        target = self.targets[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Return tuple compatible with standard training loops
        return image, target


def get_transforms(split="train"):
    """
    Returns the data transformation pipeline.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    if split == "train":
        return transforms.Compose(
            [
                # ToTensor converts (H, W, C) [0, 255] -> (C, H, W) [0.0, 1.0]
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    else:
        # Validation and Test
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


def _load_and_cache_data(
    metadata_path, cache_prefix, load_cached_data, debug, debug_size
):
    """
    Loads data from metadata CSV and images, utilizing caching to speed up subsequent runs.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cache files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, load only a subset of data.
        debug_size (int): Number of samples to load in debug mode.

    Returns:
        tuple: (images, targets, ids) as numpy arrays.
    """
    # Adjust cache prefix for debug mode to avoid overwriting full cache
    if debug:
        cache_prefix = f"{cache_prefix}_debug"

    # Define cache file paths
    images_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_images.npy")
    targets_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_targets.npy")

    # 1. Try loading from cache
    if (
        load_cached_data
        and os.path.exists(images_cache_path)
        and os.path.exists(targets_cache_path)
    ):
        try:
            images = np.load(images_cache_path)
            targets = np.load(targets_cache_path)

            # Load IDs from CSV (fast enough, avoids pickle issues with string arrays in npy)
            df = pd.read_csv(metadata_path)
            if debug:
                df = df.head(debug_size)

            # Verification: Check if cache size matches current request
            if len(images) == len(df):
                return images, targets, df["id"].values
        except Exception:
            # If cache load fails, proceed to recompute
            pass

    # 2. Compute from scratch
    df = pd.read_csv(metadata_path)
    if debug:
        df = df.head(debug_size)

    image_list = []

    for _, row in df.iterrows():
        # Construct full path: INPUT_DIR + relative_path_from_metadata
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        # Load image with OpenCV
        img = cv2.imread(full_path)

        if img is None:
            # Fallback for missing images (should not happen based on metadata verification)
            img = np.zeros((32, 32, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV default) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_list.append(img)

    # Stack into numpy arrays
    images = np.array(image_list, dtype=np.uint8)
    targets = df["has_cactus"].values.astype(np.float32)
    ids = df["id"].values

    # 3. Save to cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(images_cache_path, images)
    np.save(targets_cache_path, targets)

    return images, targets, ids


def get_datasets(load_cached_data=True, debug=DEBUG, debug_size=DEBUG_SAMPLE_SIZE):
    """
    Factory function to create Train, Validation, and Test datasets.

    Args:
        load_cached_data (bool): Whether to use cached data.
        debug (bool): Whether to run in debug mode (subset of data).
        debug_size (int): Size of the subset in debug mode.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """

    # Initialize Train Dataset
    train_imgs, train_targets, train_ids = _load_and_cache_data(
        TRAIN_METADATA_PATH, "train", load_cached_data, debug, debug_size
    )
    train_dataset = CactusDataset(
        train_imgs, train_targets, train_ids, transform=get_transforms("train")
    )

    # Initialize Validation Dataset
    val_imgs, val_targets, val_ids = _load_and_cache_data(
        VAL_METADATA_PATH, "val", load_cached_data, debug, debug_size
    )
    val_dataset = CactusDataset(
        val_imgs, val_targets, val_ids, transform=get_transforms("val")
    )

    # Initialize Test Dataset
    test_imgs, test_targets, test_ids = _load_and_cache_data(
        TEST_METADATA_PATH, "test", load_cached_data, debug, debug_size
    )
    test_dataset = CactusDataset(
        test_imgs, test_targets, test_ids, transform=get_transforms("test")
    )

    return train_dataset, val_dataset, test_dataset
