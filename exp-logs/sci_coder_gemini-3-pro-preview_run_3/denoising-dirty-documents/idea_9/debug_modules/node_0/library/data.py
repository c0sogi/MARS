import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class DenoisingDataset(Dataset):
    """
    PyTorch Dataset for the Denoising Task.
    Handles accessing patches and applying geometric augmentations.
    """

    def __init__(self, patches, targets, augment=False):
        """
        Args:
            patches (np.ndarray): Array of noisy input patches (N, H, W).
            targets (np.ndarray): Array of clean target patches (N, H, W).
            augment (bool): Whether to apply geometric augmentations.
        """
        self.patches = patches
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Retrieve patch and target
        # Copying ensures negative strides from flip/rot operations don't cause torch errors
        input_patch = self.patches[idx].copy()
        target_patch = self.targets[idx].copy()

        # Apply augmentations if enabled
        if self.augment:
            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                input_patch = np.fliplr(input_patch)
                target_patch = np.fliplr(target_patch)

            # Random Vertical Flip
            if np.random.rand() > 0.5:
                input_patch = np.flipud(input_patch)
                target_patch = np.flipud(target_patch)

            # Random 90-degree Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                input_patch = np.rot90(input_patch, k)
                target_patch = np.rot90(target_patch, k)

        # Convert to Tensor (C, H, W)
        # Input is (H, W), so we add channel dimension
        input_tensor = (
            torch.from_numpy(np.ascontiguousarray(input_patch)).float().unsqueeze(0)
        )
        target_tensor = (
            torch.from_numpy(np.ascontiguousarray(target_patch)).float().unsqueeze(0)
        )

        return input_tensor, target_tensor


def extract_patches(
    metadata_path,
    patches_cache_path,
    targets_cache_path,
    patch_size=Config.PATCH_SIZE,
    stride=Config.STRIDE,
    load_cached_data=True,
    debug=Config.DEBUG,
    debug_size=Config.DEBUG_SUBSET_SIZE,
):
    """
    Extracts high-density overlapping patches from images listed in metadata.
    Implements caching to avoid re-processing.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(patches_cache_path), exist_ok=True)

    # 1. Check Cache
    if (
        load_cached_data
        and os.path.exists(patches_cache_path)
        and os.path.exists(targets_cache_path)
    ):
        print(f"Loading cached patches from {patches_cache_path}...")
        patches = np.load(patches_cache_path)
        targets = np.load(targets_cache_path)
        return patches, targets

    print(f"Processing data from {metadata_path}...")

    # 2. Load Metadata
    df = pd.read_csv(metadata_path)

    input_patches_list = []
    target_patches_list = []

    # Iterate over images
    for idx, row in df.iterrows():
        # Debugging limit
        if (
            debug
            and len(input_patches_list)
            * (1 if not input_patches_list else len(input_patches_list[0]))
            > debug_size
        ):
            break

        input_path = os.path.join(Config.INPUT_DIR, row["input_path"])
        target_path = os.path.join(Config.INPUT_DIR, row["target_path"])

        # Load images in grayscale
        img_in = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        img_tar = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        if img_in is None or img_tar is None:
            print(f"Warning: Failed to load {row['image_id']}. Skipping.")
            continue

        # Normalize to [0, 1]
        img_in = img_in.astype(np.float32) / 255.0
        img_tar = img_tar.astype(np.float32) / 255.0

        h, w = img_in.shape

        # Extract patches
        # Loop with stride
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch_in = img_in[y : y + patch_size, x : x + patch_size]
                patch_tar = img_tar[y : y + patch_size, x : x + patch_size]

                input_patches_list.append(patch_in)
                target_patches_list.append(patch_tar)

    # Convert to numpy arrays
    print("Stacking patches...")
    patches_array = np.array(input_patches_list, dtype=np.float32)
    targets_array = np.array(target_patches_list, dtype=np.float32)

    # 3. Save to Cache
    print(f"Saving {patches_array.shape[0]} patches to cache...")
    np.save(patches_cache_path, patches_array)
    np.save(targets_cache_path, targets_array)

    return patches_array, targets_array


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Constructs and returns the training and validation DataLoaders.
    """
    set_seed(Config.SEED)

    # --- Prepare Training Data ---
    print("Preparing Training Data...")
    train_patches, train_targets = extract_patches(
        metadata_path=Config.TRAIN_METADATA_PATH,
        patches_cache_path=Config.TRAIN_PATCHES_CACHE,
        targets_cache_path=Config.TRAIN_TARGETS_CACHE,
        load_cached_data=load_cached_data,
    )

    # --- Prepare Validation Data ---
    print("Preparing Validation Data...")
    val_patches, val_targets = extract_patches(
        metadata_path=Config.VAL_METADATA_PATH,
        patches_cache_path=Config.VAL_PATCHES_CACHE,
        targets_cache_path=Config.VAL_TARGETS_CACHE,
        load_cached_data=load_cached_data,
    )

    # --- Create Datasets ---
    # Augmentation only for training
    train_dataset = DenoisingDataset(train_patches, train_targets, augment=True)
    val_dataset = DenoisingDataset(val_patches, val_targets, augment=False)

    print(f"Training Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader
