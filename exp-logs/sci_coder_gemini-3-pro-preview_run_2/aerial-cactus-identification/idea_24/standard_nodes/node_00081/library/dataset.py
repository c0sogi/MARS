import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library import config


def get_transforms(phase: str):
    """
    Returns the albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    # Base normalization to [0, 1]
    # (pixel / 255.0 - mean) / std with mean=0, std=1 -> pixel / 255.0
    base_transforms = [
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2(),
    ]

    if phase == "train":
        # Light augmentation strategy from config
        aug_params = config.AUGMENTATION
        augmentations = []

        if aug_params.get("horizontal_flip_prob", 0) > 0:
            augmentations.append(A.HorizontalFlip(p=aug_params["horizontal_flip_prob"]))

        if aug_params.get("vertical_flip_prob", 0) > 0:
            augmentations.append(A.VerticalFlip(p=aug_params["vertical_flip_prob"]))

        # Combine augmentations with base transforms
        return A.Compose(augmentations + base_transforms)

    else:
        # Validation and Test: Only normalization and tensor conversion
        return A.Compose(base_transforms)


class CactusDataset(Dataset):
    """
    Dataset class for Cactus Identification.
    Handles loading, caching, and transforming images.
    """

    def __init__(self, metadata_path, phase, transform=None, load_cached_data=True):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            phase (str): 'train', 'val', or 'test'.
            transform (callable, optional): Albumentations transform pipeline.
            load_cached_data (bool): Whether to try loading data from cache.
        """
        self.metadata_path = metadata_path
        self.phase = phase
        self.transform = transform
        self.working_dir = config.WORKING_DIR

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # Load metadata dataframe
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        self.df = pd.read_csv(metadata_path)

        # Load data (either from cache or raw files)
        self.images, self.labels, self.ids = self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        """
        Internal method to handle data loading and caching logic.
        """
        # Define cache file paths
        img_cache_path = os.path.join(self.working_dir, f"{self.phase}_images.npy")
        lbl_cache_path = os.path.join(self.working_dir, f"{self.phase}_labels.npy")
        id_cache_path = os.path.join(self.working_dir, f"{self.phase}_ids.npy")

        # 1. Try to load from cache
        if load_cached_data:
            if (
                os.path.exists(img_cache_path)
                and os.path.exists(lbl_cache_path)
                and os.path.exists(id_cache_path)
            ):

                try:
                    images = np.load(img_cache_path)
                    labels = np.load(lbl_cache_path)
                    ids = np.load(id_cache_path)
                    return images, labels, ids
                except Exception as e:
                    print(f"Failed to load cache for {self.phase}: {e}. Recomputing...")

        # 2. Compute from scratch
        images = []
        labels = []
        ids = []

        # Iterate over metadata
        for _, row in self.df.iterrows():
            # Construct full path
            # Metadata contains relative path (e.g., 'train/id.jpg')
            # Input dir is './input'
            full_path = os.path.join(config.INPUT_DIR, row["file_path"])

            # Load image
            img = cv2.imread(full_path)
            if img is None:
                # In a real scenario, we might log this.
                # Given the verified metadata, this shouldn't happen.
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            labels.append(row["has_cactus"])
            ids.append(row["id"])

        # Convert to numpy arrays
        images = np.array(images, dtype=np.uint8)
        # Labels as float32 for BCEWithLogitsLoss
        labels = np.array(labels, dtype=np.float32)
        ids = np.array(ids)

        # 3. Save to cache
        np.save(img_cache_path, images)
        np.save(lbl_cache_path, labels)
        np.save(id_cache_path, ids)

        return images, labels, ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image_id = self.ids[idx]

        if self.transform:
            # Albumentations expects image as keyword argument
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback manual transform if none provided (sanity check)
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1))

        return image, label, image_id
