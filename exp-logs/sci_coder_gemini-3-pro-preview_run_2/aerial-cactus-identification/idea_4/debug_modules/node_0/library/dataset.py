import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library import config


def get_transforms(split: str):
    """
    Returns the transformation pipeline for the given data split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transforms.
    """
    if split == "train" and config.USE_LIGHT_AUGMENTATION:
        # Light augmentation: Random Horizontal and Vertical Flips
        # ToTensor converts [0, 255] -> [0.0, 1.0]
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    else:
        # Validation/Test: No augmentation, just tensor conversion
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


def load_data(metadata_path, cache_prefix, load_cached_data=True, sample_size=None):
    """
    Loads dataset images, labels, and IDs. Implements caching for images to optimize runtime.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Identifier for the cache file (e.g., 'train', 'val').
        load_cached_data (bool): Whether to use cached data if available.
        sample_size (int, optional): Number of samples to load (for debugging).

    Returns:
        tuple: (images, labels, ids)
            images: np.ndarray of shape (N, 32, 32, 3)
            labels: np.ndarray of shape (N,)
            ids: np.ndarray of shape (N,)
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(config.WORKING_DIR, f"{cache_prefix}_images.npy")

    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Handle debugging/sampling
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        # Disable loading from cache when sampling to avoid shape mismatches
        load_cached_data = False

    images = None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cached_images = np.load(cache_path)
            # Verify consistency with metadata
            if len(cached_images) == len(df):
                images = cached_images
            else:
                # Cache mismatch (e.g. metadata changed), reload from source
                images = None
        except Exception:
            # Corrupt cache, reload from source
            images = None

    # 2. Process from scratch if needed
    if images is None:
        img_list = []
        for _, row in df.iterrows():
            # Construct full path. Metadata paths are relative to INPUT_DIR.
            # e.g., "train/0004be2cfeaba1c0361d39e2b000257b.jpg"
            full_path = os.path.join(config.INPUT_DIR, row["file_path"])

            # Read image
            img = cv2.imread(full_path)

            if img is None:
                # Fallback for safety, though metadata should be valid
                img = np.zeros(
                    (config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8
                )
            else:
                # Convert BGR (OpenCV default) to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_list.append(img)

        images = np.array(img_list, dtype=np.uint8)

        # Save to cache (only if not debugging)
        if load_cached_data and sample_size is None:
            np.save(cache_path, images)

    # Extract labels and IDs
    labels = df["has_cactus"].values.astype(np.float32)
    ids = df["id"].values

    return images, labels, ids


class CactusDataset(Dataset):
    """
    PyTorch Dataset for Cactus Classification.
    """

    def __init__(self, images, labels, ids, transform=None):
        """
        Args:
            images (np.ndarray): Image data (N, H, W, C).
            labels (np.ndarray): Target labels (N,).
            ids (np.ndarray): Image IDs (N,).
            transform (callable, optional): Transformations to apply.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
