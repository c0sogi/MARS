import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def extract_patches(image, patch_size, stride):
    """
    Extracts patches from an image using a sliding window.

    Args:
        image (np.ndarray): Input image of shape (H, W) or (H, W, C).
        patch_size (int): Size of the square patch.
        stride (int): Step size for the sliding window.

    Returns:
        np.ndarray: Array of patches of shape (N, patch_size, patch_size, C).
    """
    # Ensure image has a channel dimension
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]

    h, w, c = image.shape
    patches = []

    # Sliding window
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size, :]
            patches.append(patch)

    return np.array(patches)


def prepare_dataset_split(
    metadata_path, cache_patches_path, cache_targets_path, load_cached_data
):
    """
    Loads data from metadata, extracts patches, and caches them.
    Or loads directly from cache if available.
    """
    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(cache_patches_path)
        and os.path.exists(cache_targets_path)
    ):
        print(f"Loading cached patches from {cache_patches_path}...")
        patches = np.load(cache_patches_path)
        targets = np.load(cache_targets_path)
        return patches, targets

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    all_patches = []
    all_targets = []

    for _, row in df.iterrows():
        input_path = os.path.join(Config.INPUT_DIR, row["input_path"])
        target_path = os.path.join(Config.INPUT_DIR, row["target_path"])

        # Read images in grayscale
        img_in = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        img_tar = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        if img_in is None or img_tar is None:
            continue

        # Normalize to [0, 1]
        img_in = img_in.astype(np.float32) / 255.0
        img_tar = img_tar.astype(np.float32) / 255.0

        # Extract patches
        patches_in = extract_patches(img_in, Config.PATCH_SIZE, Config.STRIDE)
        patches_tar = extract_patches(img_tar, Config.PATCH_SIZE, Config.STRIDE)

        all_patches.append(patches_in)
        all_targets.append(patches_tar)

    # Concatenate all patches
    if all_patches:
        final_patches = np.concatenate(all_patches, axis=0)
        final_targets = np.concatenate(all_targets, axis=0)
    else:
        # Fallback for empty dataset
        final_patches = np.zeros(
            (0, Config.PATCH_SIZE, Config.PATCH_SIZE, 1), dtype=np.float32
        )
        final_targets = np.zeros(
            (0, Config.PATCH_SIZE, Config.PATCH_SIZE, 1), dtype=np.float32
        )

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_patches_path), exist_ok=True)
    np.save(cache_patches_path, final_patches)
    np.save(cache_targets_path, final_targets)
    print(f"Saved {len(final_patches)} patches to cache.")

    return final_patches, final_targets


class DenoisingDataset(Dataset):
    """
    PyTorch Dataset for Denoising Task.
    Handles on-the-fly augmentation and tensor conversion.
    """

    def __init__(self, patches, targets, augment=False):
        self.patches = patches
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Retrieve patch pair (H, W, C)
        input_patch = self.patches[idx]
        target_patch = self.targets[idx]

        # Apply Augmentation (Synchronized for Input and Target)
        if self.augment:
            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                input_patch = np.flip(input_patch, axis=1)  # Flip W
                target_patch = np.flip(target_patch, axis=1)

            # Random Vertical Flip
            if np.random.rand() > 0.5:
                input_patch = np.flip(input_patch, axis=0)  # Flip H
                target_patch = np.flip(target_patch, axis=0)

            # Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                input_patch = np.rot90(input_patch, k, axes=(0, 1))
                target_patch = np.rot90(target_patch, k, axes=(0, 1))

        # Convert to Tensor (C, H, W)
        # Input is currently (H, W, C), PyTorch expects (C, H, W)
        # Ensure memory is contiguous after numpy flips/rotations
        input_tensor = (
            torch.from_numpy(np.ascontiguousarray(input_patch)).permute(2, 0, 1).float()
        )
        target_tensor = (
            torch.from_numpy(np.ascontiguousarray(target_patch))
            .permute(2, 0, 1)
            .float()
        )

        return input_tensor, target_tensor


def get_dataloaders(load_cached_data=True):
    """
    Prepares datasets and returns DataLoaders for training and validation.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed patches from disk.

    Returns:
        train_loader, val_loader
    """
    seed_everything(Config.SEED)

    # --- Prepare Training Data ---
    train_patches, train_targets = prepare_dataset_split(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_patches_path=Config.CACHE_TRAIN_PATCHES,
        cache_targets_path=Config.CACHE_TRAIN_TARGETS,
        load_cached_data=load_cached_data,
    )

    # --- Prepare Validation Data ---
    val_patches, val_targets = prepare_dataset_split(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_patches_path=Config.CACHE_VAL_PATCHES,
        cache_targets_path=Config.CACHE_VAL_TARGETS,
        load_cached_data=load_cached_data,
    )

    # --- Instantiate Datasets ---
    train_dataset = DenoisingDataset(
        train_patches, train_targets, augment=Config.AUGMENTATION
    )

    val_dataset = DenoisingDataset(
        val_patches, val_targets, augment=False  # No augmentation for validation
    )

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        drop_last=True,  # Drop incomplete batches to maintain stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        drop_last=False,
    )

    print(f"DataLoaders prepared:")
    print(f"  Train: {len(train_dataset)} patches ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_dataset)} patches ({len(val_loader)} batches)")

    return train_loader, val_loader
