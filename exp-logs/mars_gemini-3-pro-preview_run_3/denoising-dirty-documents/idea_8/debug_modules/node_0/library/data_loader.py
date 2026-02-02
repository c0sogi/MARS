import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def load_image(path):
    """
    Loads an image from disk, converts to grayscale, and normalizes to [0, 1].
    """
    # Construct full path. Metadata contains relative paths (e.g., 'train/101.png')
    full_path = os.path.join(Config.INPUT_DIR, path)

    # Load as grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {full_path}")

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    return img


def extract_patches(metadata_path, patch_size, stride):
    """
    Loads images based on metadata and extracts overlapping patches.
    Returns noisy patches and the corresponding noise residuals (Noisy - Clean).
    """
    df = pd.read_csv(metadata_path)

    patches_noisy = []
    patches_residual = []

    for _, row in df.iterrows():
        # Load images
        img_noisy = load_image(row["input_path"])
        img_clean = load_image(row["target_path"])

        # Compute true noise residual (Target for DnCNN)
        # Noise = Noisy - Clean
        img_residual = img_noisy - img_clean

        h, w = img_noisy.shape

        # Extract patches
        # Note: We stop when the window extends beyond the image
        for r in range(0, h - patch_size + 1, stride):
            for c in range(0, w - patch_size + 1, stride):
                p_noisy = img_noisy[r : r + patch_size, c : c + patch_size]
                p_residual = img_residual[r : r + patch_size, c : c + patch_size]

                patches_noisy.append(p_noisy)
                patches_residual.append(p_residual)

    return np.array(patches_noisy), np.array(patches_residual)


def prepare_data(load_cached_data=True):
    """
    Manages data preparation and caching.
    Returns tuple: ((train_patches, train_residuals), (val_patches, val_residuals))
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define paths
    train_patches_path = Config.TRAIN_PATCHES_PATH
    train_targets_path = Config.TRAIN_TARGETS_PATH
    val_patches_path = Config.VAL_PATCHES_PATH
    val_targets_path = Config.VAL_TARGETS_PATH

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_patches_path)
        and os.path.exists(train_targets_path)
        and os.path.exists(val_patches_path)
        and os.path.exists(val_targets_path)
    )

    if load_cached_data and cache_exists:
        print("Loading cached patches from disk...")
        train_patches = np.load(train_patches_path)
        train_residuals = np.load(train_targets_path)
        val_patches = np.load(val_patches_path)
        val_residuals = np.load(val_targets_path)
    else:
        print("Processing data and extracting patches (this may take a while)...")

        # Process Train
        print(f"Extracting training patches (Stride={Config.STRIDE})...")
        train_patches, train_residuals = extract_patches(
            Config.TRAIN_CSV, Config.PATCH_SIZE, Config.STRIDE
        )

        # Process Val
        # We use the same stride for validation to get dense evaluation metrics during training,
        # or we could use a larger stride. Following Config logic, we use the same parameters.
        print(f"Extracting validation patches...")
        val_patches, val_residuals = extract_patches(
            Config.VAL_CSV, Config.PATCH_SIZE, Config.STRIDE
        )

        # Save to cache
        print("Saving processed patches to disk...")
        np.save(train_patches_path, train_patches)
        np.save(train_targets_path, train_residuals)
        np.save(val_patches_path, val_patches)
        np.save(val_targets_path, val_residuals)

    print(
        f"Data Prepared. Train shape: {train_patches.shape}, Val shape: {val_patches.shape}"
    )
    return (train_patches, train_residuals), (val_patches, val_residuals)


class DenoisingDataset(Dataset):
    """
    PyTorch Dataset for serving image patches.
    Handles channel expansion and geometric augmentations.
    """

    def __init__(self, patches, residuals, augment=False):
        """
        Args:
            patches (np.ndarray): Input noisy patches (N, H, W).
            residuals (np.ndarray): Target noise residuals (N, H, W).
            augment (bool): Whether to apply geometric augmentations.
        """
        self.patches = patches
        self.residuals = residuals
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Get data (H, W)
        patch = self.patches[idx]
        residual = self.residuals[idx]

        # Add channel dimension (H, W) -> (C, H, W) where C=1
        # We do this before augmentation to handle axes consistently
        patch = patch[np.newaxis, :, :]
        residual = residual[np.newaxis, :, :]

        if self.augment:
            # 1. Random Horizontal/Vertical Flips
            # Axis 1 is H, Axis 2 is W
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=2)
                residual = np.flip(residual, axis=2)

            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=1)
                residual = np.flip(residual, axis=1)

            # 2. Random 90-degree Rotations
            k = np.random.randint(0, 4)
            if k > 0:
                # rot90 rotates in the plane defined by axes.
                patch = np.rot90(patch, k, axes=(1, 2))
                residual = np.rot90(residual, k, axes=(1, 2))

        # Convert to torch tensors
        # Copy is required because numpy flips/rotations return negative strides
        # which torch doesn't support directly.
        patch_tensor = torch.from_numpy(patch.copy()).float()
        residual_tensor = torch.from_numpy(residual.copy()).float()

        return patch_tensor, residual_tensor


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns training and validation DataLoaders.

    Args:
        load_cached_data (bool): Whether to attempt loading from .npy cache.

    Returns:
        train_loader, val_loader
    """
    # 1. Prepare Data
    (train_x, train_y), (val_x, val_y) = prepare_data(load_cached_data=load_cached_data)

    # 2. Create Datasets
    # Augmentation only for training
    train_dataset = DenoisingDataset(train_x, train_y, augment=Config.AUGMENTATION)
    val_dataset = DenoisingDataset(val_x, val_y, augment=False)

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader
