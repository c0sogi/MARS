import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config


class CactusDataset(Dataset):
    """
    PyTorch Dataset for the Cactus Aerial Photos.
    Handles image conversion and augmentation.
    """

    def __init__(self, images, targets, transform=None, is_test=False):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C) in uint8.
            targets (np.ndarray): Array of labels (int) or IDs (str).
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): If True, returns (image, id). If False, returns (image, label).
        """
        self.images = images
        self.targets = targets
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image and target
        img = self.images[idx]  # HWC, uint8
        target = self.targets[idx]

        # Apply transformations (Augmentation + Normalization to [0, 1] + Channel First)
        if self.transform:
            img = self.transform(img)
        else:
            # Fallback if no transform provided: Convert to Tensor (CHW, 0-1)
            img = transforms.functional.to_tensor(img)

        if self.is_test:
            # For test set, return the image ID to map predictions later
            return img, target
        else:
            # For train/val, return the label as float32 for BCEWithLogitsLoss
            return img, torch.tensor(target, dtype=torch.float32)


def _load_data(metadata_path, cache_prefix, load_cached_data):
    """
    Loads data from metadata CSV and images from disk, with caching.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images_array, targets_array)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    img_cache_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_images.npy")
    target_cache_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_targets.npy")

    # 1. Try loading from cache
    if (
        load_cached_data
        and os.path.exists(img_cache_path)
        and os.path.exists(target_cache_path)
    ):
        try:
            images = np.load(img_cache_path)
            # allow_pickle=True is required for string arrays (IDs in test set)
            targets = np.load(target_cache_path, allow_pickle=True)
            return images, targets
        except Exception:
            # If load fails, fall through to processing from scratch
            pass

    # 2. Process from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    images_list = []
    targets_list = []

    for _, row in df.iterrows():
        # Metadata contains relative paths, e.g., "train/xxx.jpg"
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image using OpenCV
        img = cv2.imread(full_path)
        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_list.append(img)

        # Collect targets
        if "test" in cache_prefix:
            targets_list.append(row["id"])
        else:
            targets_list.append(row["has_cactus"])

    # Convert to numpy arrays
    images = np.array(images_list, dtype=np.uint8)
    targets = np.array(targets_list)  # int64 for labels, object/str for IDs

    # 3. Save to cache
    np.save(img_cache_path, images)
    np.save(target_cache_path, targets)

    return images, targets


def get_loaders(load_cached_data: bool = True):
    """
    Constructs and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npy files if available.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # --- 1. Load Data ---
    train_imgs, train_labels = _load_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_labels = _load_data(Config.VAL_METADATA_PATH, "val", load_cached_data)
    test_imgs, test_ids = _load_data(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    # --- 2. Debug Mode ---
    if Config.DEBUG:
        limit = min(len(train_imgs), Config.DEBUG_SAMPLE_SIZE)
        train_imgs = train_imgs[:limit]
        train_labels = train_labels[:limit]

        limit_val = min(len(val_imgs), Config.DEBUG_SAMPLE_SIZE)
        val_imgs = val_imgs[:limit_val]
        val_labels = val_labels[:limit_val]

        limit_test = min(len(test_imgs), Config.DEBUG_SAMPLE_SIZE)
        test_imgs = test_imgs[:limit_test]
        test_ids = test_ids[:limit_test]

    # --- 3. Define Transformations ---
    # Train: Light Augmentation + Normalization
    # ToPILImage needed because input is numpy array
    train_ops = [transforms.ToPILImage()]

    if Config.AUG_HORIZONTAL_FLIP:
        train_ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if Config.AUG_VERTICAL_FLIP:
        train_ops.append(transforms.RandomVerticalFlip(p=0.5))

    train_ops.append(transforms.ToTensor())  # Converts to [0, 1]

    train_transform = transforms.Compose(train_ops)

    # Eval: Normalization only
    eval_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.ToTensor()]
    )

    # --- 4. Instantiate Datasets ---
    train_dataset = CactusDataset(
        train_imgs, train_labels, transform=train_transform, is_test=False
    )
    val_dataset = CactusDataset(
        val_imgs, val_labels, transform=eval_transform, is_test=False
    )
    test_dataset = CactusDataset(
        test_imgs, test_ids, transform=eval_transform, is_test=True
    )

    # --- 5. Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
