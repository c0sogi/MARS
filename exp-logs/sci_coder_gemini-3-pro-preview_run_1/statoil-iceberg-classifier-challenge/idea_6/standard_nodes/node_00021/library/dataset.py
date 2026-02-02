import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_logger

logger = get_logger("Dataset")


def load_and_cache_data(metadata_path, json_path, cache_prefix, load_cached_data=True):
    """
    Loads data from JSON based on metadata, processes it into numpy arrays,
    and caches it to disk to speed up subsequent loads.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        json_path (str): Path to the raw JSON data file.
        cache_prefix (str): Prefix for the cached .npy files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, angles, labels, ids)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    p_imgs = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    p_angs = os.path.join(cache_dir, f"{cache_prefix}_angles.npy")
    p_lbls = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")
    p_ids = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    has_labels = "test" not in cache_prefix

    # Check if all required cache files exist
    cache_exists = (
        os.path.exists(p_imgs) and os.path.exists(p_angs) and os.path.exists(p_ids)
    )
    if has_labels:
        cache_exists = cache_exists and os.path.exists(p_lbls)

    if load_cached_data and cache_exists:
        logger.info(f"Loading cached data for '{cache_prefix}' from {cache_dir}...")
        images = np.load(p_imgs)
        angles = np.load(p_angs)
        ids = np.load(p_ids)
        labels = np.load(p_lbls) if has_labels else None
        return images, angles, labels, ids

    logger.info(
        f"Cache miss or reload requested. Processing '{cache_prefix}' from scratch..."
    )

    # Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    df = pd.read_csv(metadata_path)

    # Load Raw JSON
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Map metadata to raw data using sample_index
    indices = df["sample_index"].values
    samples = [raw_data[i] for i in indices]

    # Process Images: Band 1, Band 2 -> Band 3 (Mean) -> Stack
    n_samples = len(samples)
    images = np.zeros((n_samples, 75, 75, 3), dtype=np.float32)

    for i, s in enumerate(samples):
        b1 = np.array(s["band_1"], dtype=np.float32).reshape(75, 75)
        b2 = np.array(s["band_2"], dtype=np.float32).reshape(75, 75)
        b3 = (b1 + b2) / 2.0
        images[i, :, :, 0] = b1
        images[i, :, :, 1] = b2
        images[i, :, :, 2] = b3

    # Process Angles: Handle 'na' and impute
    angles = []
    for s in samples:
        a = s["inc_angle"]
        if isinstance(a, str) and a.lower() == "na":
            angles.append(np.nan)
        else:
            angles.append(float(a))
    angles = np.array(angles, dtype=np.float32)

    # Impute missing angles with mean of the current split
    valid_mask = ~np.isnan(angles)
    if np.sum(valid_mask) > 0:
        mean_angle = np.mean(angles[valid_mask])
        angles[np.isnan(angles)] = mean_angle
    else:
        angles[:] = 0.0  # Fallback

    # Process IDs
    ids = np.array([s["id"] for s in samples])

    # Process Labels
    labels = None
    if has_labels:
        labels = np.array([s["is_iceberg"] for s in samples], dtype=np.float32)
        np.save(p_lbls, labels)

    # Save Cache
    np.save(p_imgs, images)
    np.save(p_angs, angles)
    np.save(p_ids, ids)

    return images, angles, labels, ids


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load base image (75, 75, 3)
        image = self.images[idx].copy()
        angle = self.angles[idx]
        sample_id = self.ids[idx]

        # 1. Per-Sample Min-Max Scaling
        # Normalize to [0, 1] based on local min/max
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val > 1e-6:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = image - min_val  # Zero center if flat

        # 2. Upsampling
        # Resize to (224, 224) using bicubic interpolation
        image = cv2.resize(
            image, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_CUBIC
        )

        # 3. Augmentation
        if self.transform:
            # Albumentations expects HWC numpy array
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Convert to Tensor (C, H, W)
            image = torch.from_numpy(image).float().permute(2, 0, 1)

        # Convert angle to tensor
        angle = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.float32)
            return image, angle, label, sample_id
        else:
            return image, angle, sample_id


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for train/val/test.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_dataset(split, load_cached_data=True):
    """
    Factory function to create datasets for specific splits.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        IcebergDataset: The constructed dataset.
    """
    if split == "train":
        meta_path = Config.TRAIN_META_PATH
        json_path = Config.TRAIN_JSON
        cache_prefix = "train"
        transform = get_transforms("train")
    elif split == "val":
        meta_path = Config.VAL_META_PATH
        json_path = Config.TRAIN_JSON  # Val subset is in train.json
        cache_prefix = "val"
        transform = get_transforms("val")
    elif split == "test":
        meta_path = Config.TEST_META_PATH
        json_path = Config.TEST_JSON
        cache_prefix = "test"
        transform = get_transforms("test")
    else:
        raise ValueError(f"Unknown split: {split}")

    images, angles, labels, ids = load_and_cache_data(
        meta_path, json_path, cache_prefix, load_cached_data
    )

    # Handle Debugging: Reduce dataset size if DEBUG is enabled
    if Config.DEBUG:
        size = min(len(images), Config.DEBUG_SAMPLE_SIZE)
        logger.info(f"DEBUG mode: Truncating {split} dataset to {size} samples.")
        images = images[:size]
        angles = angles[:size]
        ids = ids[:size]
        if labels is not None:
            labels = labels[:size]

    dataset = IcebergDataset(
        images=images, angles=angles, labels=labels, ids=ids, transform=transform
    )

    return dataset
