import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config


def get_transforms(phase: str):
    """
    Returns the data transformations for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The composition of transforms.
    """
    if phase == "train":
        # Training: H-Flip, V-Flip, then ToTensor
        # ToPILImage is used first to ensure compatibility with torchvision transforms if input is numpy
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=Config.AUG_H_FLIP_PROB),
                transforms.RandomVerticalFlip(p=Config.AUG_V_FLIP_PROB),
                transforms.ToTensor(),
            ]
        )
    else:
        # Validation/Test: Just convert to Tensor (HWC [0,255] -> CHW [0.0, 1.0])
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


def load_and_cache_data(
    metadata_path: str, cache_prefix: str, load_cached_data: bool = True
):
    """
    Loads data from metadata and images, caching the result as .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cached files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (ids, images, labels) as numpy arrays.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    ids_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")
    images_path = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    labels_path = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(ids_path)
            and os.path.exists(images_path)
            and os.path.exists(labels_path)
        ):
            try:
                # Allow pickle for object arrays (ids are strings)
                ids = np.load(ids_path, allow_pickle=True)
                images = np.load(images_path)
                labels = np.load(labels_path)
                return ids, images, labels
            except Exception:
                # If loading fails, proceed to recompute
                pass

    # 2. Compute from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    ids_list = []
    images_list = []
    labels_list = []

    input_dir = Config.INPUT_DIR

    for _, row in df.iterrows():
        img_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        # Load image using OpenCV
        img = cv2.imread(full_path)
        if img is None:
            continue

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ids_list.append(img_id)
        images_list.append(img)
        labels_list.append(row["has_cactus"])

    ids = np.array(ids_list)
    # Images are stored as uint8 to save space
    images = np.array(images_list, dtype=np.uint8)
    # Labels as float32 for BCEWithLogitsLoss
    labels = np.array(labels_list, dtype=np.float32)

    # 3. Save to cache
    np.save(ids_path, ids)
    np.save(images_path, images)
    np.save(labels_path, labels)

    return ids, images, labels


class CactusDataset(Dataset):
    """
    Custom Dataset for Cactus Identification.
    Loads images into memory for efficiency given the small dataset size.
    """

    def __init__(
        self,
        metadata_path: str,
        phase: str = "train",
        load_cached_data: bool = True,
        transform=None,
        limit: int = None,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV.
            phase (str): 'train', 'val', or 'test'. Used for cache naming and default transforms.
            load_cached_data (bool): Whether to use cached .npy files.
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): Limit the dataset size for debugging.
        """
        self.phase = phase
        self.transform = transform if transform is not None else get_transforms(phase)

        # Determine cache prefix based on phase
        cache_prefix = phase

        # Load data (cached or fresh)
        self.ids, self.images, self.labels = load_and_cache_data(
            metadata_path, cache_prefix, load_cached_data
        )

        # Apply limit if requested (useful for debugging)
        if limit is not None:
            self.ids = self.ids[:limit]
            self.images = self.images[:limit]
            self.labels = self.labels[:limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]
        label = self.labels[idx]
        img_id = self.ids[idx]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Return tuple: (image, label, id)
        # The training loop can unpack this as needed
        return image, label, img_id
