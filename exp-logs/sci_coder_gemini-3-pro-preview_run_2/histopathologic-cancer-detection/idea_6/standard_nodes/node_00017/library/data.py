import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for Pathology Images.
    """

    def __init__(self, images, labels=None, transforms=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray, optional): Array of labels with shape (N,).
            transforms (albumentations.Compose, optional): Augmentation pipeline.
        """
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image
        image = self.images[idx]

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Handle label
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return {
                "image": image,
                "label": label,
                "id": idx,
            }  # ID passed for tracking if needed, though not strictly required by logic
        else:
            return {"image": image}


def get_transforms(split):
    """
    Returns the augmentation pipeline based on the data split.

    Strategy:
    - Train: Global geometric & intensity augs on 96x96 -> CenterCrop 64x64 -> Normalize.
    - Val/Test: CenterCrop 64x64 -> Normalize.
    """
    mean = Config.MEAN
    std = Config.STD
    input_size = Config.INPUT_SIZE  # 64

    if split == "train":
        return A.Compose(
            [
                # Global Geometric Augmentations (on full 96x96 image)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Intensity Augmentations
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                # Contextual Crop
                A.CenterCrop(height=input_size, width=input_size),
                # Normalization & Tensor Conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # Deterministic Center Crop
                A.CenterCrop(height=input_size, width=input_size),
                # Normalization & Tensor Conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def load_data(split, load_cached_data=True):
    """
    Loads image data and labels, using caching to speed up subsequent runs.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from .npy cache.

    Returns:
        tuple: (images_array, labels_array, ids_list)
    """
    # Define cache paths
    cache_images_path = os.path.join(Config.CACHE_DIR, f"{split}_images.npy")
    cache_labels_path = os.path.join(Config.CACHE_DIR, f"{split}_labels.npy")
    # We also cache IDs to ensure alignment, though usually metadata is stable
    cache_ids_path = os.path.join(Config.CACHE_DIR, f"{split}_ids.npy")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_images_path)
            and os.path.exists(cache_labels_path)
            and os.path.exists(cache_ids_path)
        ):
            print(f"Loading {split} data from cache...")
            images = np.load(cache_images_path)
            labels = np.load(cache_labels_path)
            ids = np.load(cache_ids_path, allow_pickle=True)
            return images, labels, ids
        else:
            print(f"Cache not found for {split}. Loading from raw files...")

    # Determine metadata file
    if split == "train":
        meta_path = Config.TRAIN_META_PATH
    elif split == "val":
        meta_path = Config.VAL_META_PATH
    elif split == "test":
        meta_path = Config.TEST_META_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # Load metadata
    df = pd.read_csv(meta_path)

    # Pre-allocate arrays
    # Images are 96x96x3 uint8
    n_samples = len(df)
    images = np.zeros(
        (n_samples, Config.FULL_IMAGE_SIZE, Config.FULL_IMAGE_SIZE, 3), dtype=np.uint8
    )
    labels = np.zeros(n_samples, dtype=np.int64)
    ids = df["id"].values

    print(f"Processing {n_samples} images for {split} set...")

    # Iterate and load images
    for idx, row in df.iterrows():
        # Construct full path
        # Metadata file_path is relative, e.g., "train/xxxx.tif"
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        img = cv2.imread(full_path)

        if img is None:
            # Fallback for missing images (should not happen based on EDA)
            # Create a black image to maintain array shape
            img = np.zeros(
                (Config.FULL_IMAGE_SIZE, Config.FULL_IMAGE_SIZE, 3), dtype=np.uint8
            )
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images[idx] = img
        labels[idx] = row["label"]

    # Save to cache
    print(f"Saving {split} data to cache at {Config.CACHE_DIR}...")
    np.save(cache_images_path, images)
    np.save(cache_labels_path, labels)
    np.save(cache_ids_path, ids)

    return images, labels, ids


def get_loaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load data arrays
    train_images, train_labels, _ = load_data("train", load_cached_data)
    val_images, val_labels, _ = load_data("val", load_cached_data)
    test_images, test_labels, test_ids = load_data("test", load_cached_data)

    # Create Datasets
    train_dataset = PathologyDataset(
        images=train_images, labels=train_labels, transforms=get_transforms("train")
    )

    val_dataset = PathologyDataset(
        images=val_images, labels=val_labels, transforms=get_transforms("val")
    )

    test_dataset = PathologyDataset(
        images=test_images,
        labels=None,  # Test set labels are placeholders
        transforms=get_transforms("test"),
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
