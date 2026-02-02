import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


# --- Transforms ---
def get_transforms(split):
    """
    Returns the Albumentations composition for the given split.
    Follows the 'Augment-then-Crop' strategy defined in Config.
    """
    mean = Config.MEAN
    std = Config.STD
    crop_size = Config.CROP_SIZE

    transforms_list = []

    if split == "train":
        # Global Augmentations on 96x96
        # Geometric
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )

        # Intensity
        transforms_list.append(
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5)
        )

    # Common steps: Center Crop -> Normalize -> ToTensor
    # CenterCrop ensures we focus on the ROI + context, removing outer rim
    transforms_list.extend(
        [
            A.CenterCrop(height=crop_size, width=crop_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms_list)


# --- Data Loading & Caching ---
def load_data(split, load_cached_data=True):
    """
    Loads image data and labels.
    Uses caching (npy files) to speed up subsequent runs.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        images (np.ndarray): Array of images (N, H, W, C)
        targets (np.ndarray or list): Labels or IDs.
    """
    debug_suffix = "_debug" if Config.DEBUG else ""
    cache_dir = Config.CACHE_DIR

    # Define cache filenames
    images_cache_path = os.path.join(cache_dir, f"{split}_images{debug_suffix}.npy")
    targets_cache_path = os.path.join(cache_dir, f"{split}_labels{debug_suffix}.npy")
    # For test, we use IDs instead of labels
    ids_cache_path = os.path.join(cache_dir, f"{split}_ids{debug_suffix}.npy")

    # 1. Try loading from cache
    if load_cached_data:
        if split == "test":
            if os.path.exists(images_cache_path) and os.path.exists(ids_cache_path):
                images = np.load(images_cache_path)
                ids = np.load(ids_cache_path)
                return images, ids
        else:
            if os.path.exists(images_cache_path) and os.path.exists(targets_cache_path):
                images = np.load(images_cache_path)
                labels = np.load(targets_cache_path)
                return images, labels

    # 2. Process from scratch
    # Select metadata file
    if split == "train":
        meta_path = Config.TRAIN_META_PATH
    elif split == "val":
        meta_path = Config.VAL_META_PATH
    else:
        meta_path = Config.TEST_META_PATH

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)

    # Handle Debug Mode
    if Config.DEBUG:
        df = df.sample(
            n=min(len(df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Pre-allocate arrays
    num_samples = len(df)
    h, w = Config.RAW_IMAGE_SIZE, Config.RAW_IMAGE_SIZE
    c = 3

    images = np.zeros((num_samples, h, w, c), dtype=np.uint8)

    # Lists for targets/ids
    labels_list = []
    ids_list = []

    # Iterate and load
    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            # Fallback for missing images, create black image
            img = np.zeros((h, w, c), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images[idx] = img

        if split == "test":
            ids_list.append(row["id"])
        else:
            labels_list.append(row["label"])

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(images_cache_path, images)

    if split == "test":
        ids_array = np.array(ids_list)
        np.save(ids_cache_path, ids_array)
        return images, ids_array
    else:
        labels_array = np.array(labels_list, dtype=np.int64)
        np.save(targets_cache_path, labels_array)
        return images, labels_array


# --- Dataset Class ---
class PathologyDataset(Dataset):
    def __init__(self, images, targets, transforms=None, is_test=False):
        """
        Args:
            images (np.ndarray): (N, H, W, C) uint8
            targets (np.ndarray): (N,) int64 labels OR (N,) string IDs if is_test=True
            transforms (albumentations.Compose): Transforms to apply
            is_test (bool): If True, targets are treated as IDs.
        """
        self.images = images
        self.targets = targets
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.is_test:
            # Return image and ID
            return image, target
        else:
            # Return image and label
            return image, target


# --- DataLoader Factories ---
def get_dataloaders(load_cached_data=True):
    """
    Creates train and validation dataloaders.
    """
    # Load data
    train_images, train_labels = load_data("train", load_cached_data=load_cached_data)
    val_images, val_labels = load_data("val", load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = PathologyDataset(
        train_images, train_labels, transforms=get_transforms("train"), is_test=False
    )

    val_dataset = PathologyDataset(
        val_images, val_labels, transforms=get_transforms("val"), is_test=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Good for Batch Norm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Creates test dataloader.
    """
    test_images, test_ids = load_data("test", load_cached_data=load_cached_data)

    test_dataset = PathologyDataset(
        test_images, test_ids, transforms=get_transforms("test"), is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
