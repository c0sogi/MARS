import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import INPUT_DIR, CACHE_DIR, IMAGE_SIZE, MIXUP_ALPHA, SEED

# Statistics derived from the provided data analysis
# Mean and Std for normalization (RGB channels)
MEAN = [0.50339, 0.45197, 0.46825]
STD = [0.15138, 0.13993, 0.15355]


def load_and_cache_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads images and labels based on the provided metadata file.
    Implements caching using .npy files to speed up subsequent runs.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_prefix (str): Prefix for cache filenames (e.g., 'train', 'val').
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (ids, images, labels)
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache file paths
    ids_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")
    imgs_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_imgs.npy")
    labels_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_labels.npy")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(ids_cache_path)
            and os.path.exists(imgs_cache_path)
            and os.path.exists(labels_cache_path)
        ):
            try:
                ids = np.load(ids_cache_path, allow_pickle=True)
                images = np.load(imgs_cache_path)
                labels = np.load(labels_cache_path)
                return ids, images, labels
            except Exception:
                # If load fails, fall through to re-compute
                pass

    # Compute from scratch
    df = pd.read_csv(metadata_path)

    ids_list = []
    images_list = []
    labels_list = []

    for _, row in df.iterrows():
        img_id = row["id"]
        label = row["has_cactus"]
        rel_path = row["file_path"]

        # Construct full path
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            # Fallback for missing images (though validation says none are missing)
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1] float32
            img = img.astype(np.float32) / 255.0

        ids_list.append(img_id)
        images_list.append(img)
        labels_list.append(label)

    # Convert to numpy arrays
    ids = np.array(ids_list)
    images = np.array(images_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)

    # Save to cache
    np.save(ids_cache_path, ids)
    np.save(imgs_cache_path, images)
    np.save(labels_cache_path, labels)

    return ids, images, labels


class CactusDataset(Dataset):
    def __init__(self, ids, images, labels, transform=None):
        """
        PyTorch Dataset for Cactus images.

        Args:
            ids (np.ndarray): Array of image IDs.
            images (np.ndarray): Array of images (N, H, W, C) in float32 [0,1].
            labels (np.ndarray): Array of labels (N,).
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.ids = ids
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        img_id = self.ids[idx]

        if self.transform:
            # Albumentations works with float32 images
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Manual conversion to tensor if no transform provided
            image = torch.tensor(image).permute(2, 0, 1)

        return {
            "id": img_id,
            "image": image,
            "target": torch.tensor(label, dtype=torch.float32),
        }


def get_transforms(split="train"):
    """
    Creates the augmentation pipeline.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0), ToTensorV2()]
        )


def mixup_collate_fn(batch):
    """
    Collate function to apply Mixup regularization.

    Args:
        batch (list): List of dicts from Dataset.__getitem__.

    Returns:
        dict: Batch dictionary with mixed images and dual targets.
    """
    images = torch.stack([item["image"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    ids = [item["id"] for item in batch]

    # Determine mixup lambda
    if MIXUP_ALPHA > 0:
        lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
    else:
        lam = 1.0

    # Shuffle batch for mixing
    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    # Mix images
    mixed_images = lam * images + (1 - lam) * images[index, :]

    # Prepare dual targets
    target_a = targets
    target_b = targets[index]

    return {
        "id": ids,
        "image": mixed_images,
        "target_a": target_a,
        "target_b": target_b,
        "lam": lam,
    }
