import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from library.config import Config
from library.utils import save_array, load_array


def get_transforms(mode="train"):
    """
    Returns the data transformation pipeline based on the mode.

    Args:
        mode (str): One of 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    if mode == "train":
        # Light Augmentation: HFlip + VFlip, then Normalize to [0, 1]
        transform_list = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),  # Converts [0, 255] -> [0.0, 1.0]
        ]
    else:
        # Val/Test: Just Normalize to [0, 1]
        transform_list = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]

    return transforms.Compose(transform_list)


def _load_or_create_cache(metadata_path, cache_prefix, load_cached_data=True):
    """
    Internal helper to load data from cache or process from scratch.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_prefix (str): Prefix for the cached files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (ids, images, labels) as numpy arrays.
    """
    # Define cache paths
    cache_ids_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_ids.npy")
    cache_imgs_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_images.npy")
    cache_lbls_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_labels.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(cache_ids_path)
        and os.path.exists(cache_imgs_path)
        and os.path.exists(cache_lbls_path)
    )

    # 1. Try to load cached data
    if load_cached_data and cache_exists:
        try:
            ids = load_array(cache_ids_path)
            images = load_array(cache_imgs_path)
            labels = load_array(cache_lbls_path)
            # print(f"Loaded {cache_prefix} data from cache.")
            return ids, images, labels
        except Exception:
            # If loading fails, fall through to processing
            pass

    # 2. Process from scratch
    # print(f"Processing {cache_prefix} data from scratch...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    ids = []
    images = []
    labels = []

    # Pre-allocate lists or iterate
    # Given dataset size (~14k), lists are fine

    for _, row in df.iterrows():
        img_id = row["id"]
        label = row["has_cactus"]
        rel_path = row["file_path"]

        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load image using OpenCV
        img = cv2.imread(full_path)
        if img is None:
            raise ValueError(f"Failed to load image: {full_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Verify size (though metadata analysis confirmed 32x32)
        if img.shape[:2] != Config.IMAGE_SIZE:
            img = cv2.resize(img, Config.IMAGE_SIZE)

        ids.append(img_id)
        images.append(img)
        labels.append(label)

    # Convert to numpy arrays
    ids_np = np.array(ids)
    images_np = np.array(images, dtype=np.uint8)  # Keep as uint8 to save space
    labels_np = np.array(
        labels, dtype=np.float32
    )  # BCEWithLogitsLoss expects float targets

    # Save to cache
    save_array(ids_np, cache_ids_path)
    save_array(images_np, cache_imgs_path)
    save_array(labels_np, cache_lbls_path)

    return ids_np, images_np, labels_np


class CactusDataset(Dataset):
    """
    Dataset class for Cactus Identification.
    Loads data into memory for fast access.
    """

    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npy files.
        """
        self.mode = mode
        self.transform = get_transforms(mode)

        # Determine metadata path and cache prefix based on mode
        if mode == "train":
            meta_path = Config.TRAIN_METADATA_PATH
            prefix = "train"
        elif mode == "val":
            meta_path = Config.VAL_METADATA_PATH
            prefix = "val"
        elif mode == "test":
            meta_path = Config.TEST_METADATA_PATH
            prefix = "test"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.ids, self.images, self.labels = _load_or_create_cache(
            meta_path, prefix, load_cached_data
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Retrieve data
        img_arr = self.images[idx]  # Shape: (32, 32, 3), dtype: uint8
        label = self.labels[idx]
        img_id = self.ids[idx]

        # Apply transforms
        # Transform expects PIL Image or Tensor.
        # Since we stored as uint8 numpy, ToPILImage in transform pipeline handles it.
        image = self.transform(img_arr)

        if self.mode == "test":
            # For test, we need the ID for submission
            return image, img_id
        else:
            # For train/val, we need the label
            # Ensure label is a tensor of shape (1,) for BCE
            return image, torch.tensor(label, dtype=torch.float32)
