import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from library.config import Config, seed_everything


class MemoryPathologyDataset(Dataset):
    """
    Dataset class that holds all images in memory (RAM) as NumPy arrays.
    Applies Albumentations transforms on-the-fly.
    """

    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray, optional): Array of labels with shape (N,).
            transform (A.Compose, optional): Albumentations pipeline.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image from memory
        image = self.images[idx]

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return image and label (if available)
        if self.labels is not None:
            label = self.labels[idx]
            # BCEWithLogitsLoss expects float targets
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            return image


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline for the specified phase.
    Implements Isotropic Augmentation and Contextual Cropping.
    """
    mean = Config.normalize_mean
    std = Config.normalize_std

    if phase == "train":
        return A.Compose(
            [
                # 1. Isotropic Geometric Augmentation
                # Continuous rotation (-180 to 180) to learn orientation invariance
                A.Rotate(
                    limit=Config.aug_rotate_limit,
                    p=1.0,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                # Dihedral symmetries
                A.HorizontalFlip(p=Config.aug_flip_prob),
                A.VerticalFlip(p=Config.aug_flip_prob),
                # 2. Intensity Invariance
                # Force model to ignore stain color variations
                A.ColorJitter(
                    brightness=Config.aug_color_jitter_strength,
                    contrast=Config.aug_color_jitter_strength,
                    saturation=Config.aug_color_jitter_strength,
                    hue=Config.aug_color_jitter_strength,
                    p=Config.aug_color_jitter_prob,
                ),
                # 3. Contextual Crop
                # Crop 64x64 from the center of the augmented 96x96 patch
                A.CenterCrop(
                    height=Config.image_crop_size, width=Config.image_crop_size
                ),
                # 4. Normalization & Tensor Conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test (including TTA base)
        # Deterministic Center Crop and Normalize
        return A.Compose(
            [
                A.CenterCrop(
                    height=Config.image_crop_size, width=Config.image_crop_size
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def load_data_to_memory(load_cached_data=True):
    """
    Loads the entire dataset into RAM.
    Uses caching to speed up subsequent executions.
    Merges provided train and val metadata for full Cross-Validation.

    Args:
        load_cached_data (bool): If True, attempts to load from .npy files.

    Returns:
        tuple: (train_images, train_labels, test_images, test_ids)
    """
    # Ensure cache directory exists
    os.makedirs(Config.cache_dir, exist_ok=True)

    # Define cache file paths
    path_train_imgs = os.path.join(Config.cache_dir, "train_images.npy")
    path_train_lbls = os.path.join(Config.cache_dir, "train_labels.npy")
    path_test_imgs = os.path.join(Config.cache_dir, "test_images.npy")
    path_test_ids = os.path.join(Config.cache_dir, "test_ids.npy")

    # Check if all cache files exist
    cache_exists = (
        os.path.exists(path_train_imgs)
        and os.path.exists(path_train_lbls)
        and os.path.exists(path_test_imgs)
        and os.path.exists(path_test_ids)
    )

    if load_cached_data and cache_exists:
        print("Loading dataset from cache...")
        train_images = np.load(path_train_imgs, allow_pickle=True)
        train_labels = np.load(path_train_lbls, allow_pickle=True)
        test_images = np.load(path_test_imgs, allow_pickle=True)
        test_ids = np.load(path_test_ids, allow_pickle=True)
        return train_images, train_labels, test_images, test_ids

    print("Cache not found or disabled. Loading data from disk...")

    # --- Helper Function for Image Loading ---
    def load_images_from_metadata(df):
        count = len(df)
        # Pre-allocate memory: (N, 96, 96, 3) uint8
        imgs = np.zeros(
            (count, Config.image_raw_size, Config.image_raw_size, 3), dtype=np.uint8
        )

        for i, row in df.iterrows():
            # Construct full path
            file_path = os.path.join(Config.input_dir, row["file_path"])

            # Read image using OpenCV (BGR)
            img = cv2.imread(file_path)

            if img is not None:
                # Convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs[i] = img
            else:
                # This should not happen given metadata verification
                # Leave as zeros (black image) to maintain array shape
                pass

        return imgs

    # --- 1. Load and Merge Training Data ---
    # We combine the provided 'train' and 'val' splits to perform our own 5-Fold CV
    df_train_part = pd.read_csv(Config.train_metadata_path)
    df_val_part = pd.read_csv(Config.val_metadata_path)
    df_full_train = pd.concat([df_train_part, df_val_part], ignore_index=True)

    print(f"Loading {len(df_full_train)} training images...")
    train_images = load_images_from_metadata(df_full_train)
    train_labels = df_full_train["label"].values.astype(np.int64)

    # --- 2. Load Test Data ---
    df_test = pd.read_csv(Config.test_metadata_path)

    print(f"Loading {len(df_test)} test images...")
    test_images = load_images_from_metadata(df_test)
    test_ids = df_test["id"].values

    # --- 3. Save to Cache ---
    print("Saving dataset to cache for future runs...")
    np.save(path_train_imgs, train_images)
    np.save(path_train_lbls, train_labels)
    np.save(path_test_imgs, test_images)
    np.save(path_test_ids, test_ids)

    return train_images, train_labels, test_images, test_ids


def get_fold_dataloaders(fold_idx, train_images, train_labels):
    """
    Generates DataLoaders for a specific fold using Stratified K-Fold.

    Args:
        fold_idx (int): Index of the current fold (0 to n_folds-1).
        train_images (np.ndarray): Full training image array.
        train_labels (np.ndarray): Full training label array.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Get indices for the requested fold
    # We convert generator to list to access specific fold
    splits = list(skf.split(train_images, train_labels))
    train_indices, val_indices = splits[fold_idx]

    # Debug Mode: Subset data if enabled
    if Config.debug:
        train_indices = train_indices[: Config.debug_sample_size]
        val_indices = val_indices[: Config.debug_sample_size]

    # Create Datasets
    # Slicing numpy arrays creates views or copies, handled efficiently in memory
    train_ds = MemoryPathologyDataset(
        images=train_images[train_indices],
        labels=train_labels[train_indices],
        transform=get_transforms("train"),
    )

    val_ds = MemoryPathologyDataset(
        images=train_images[val_indices],
        labels=train_labels[val_indices],
        transform=get_transforms("val"),
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,  # Optimal for CUDA
        drop_last=True,  # Drop incomplete batches to maintain stats stability
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(test_images):
    """
    Generates DataLoader for the test set.

    Args:
        test_images (np.ndarray): Full test image array.

    Returns:
        DataLoader: Test data loader.
    """
    # Debug Mode: Subset data if enabled
    if Config.debug:
        test_images = test_images[: Config.debug_sample_size]

    test_ds = MemoryPathologyDataset(
        images=test_images, labels=None, transform=get_transforms("test")
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
