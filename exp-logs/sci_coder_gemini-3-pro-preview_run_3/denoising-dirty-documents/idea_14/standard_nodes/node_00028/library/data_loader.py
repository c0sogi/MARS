import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class DenoisingDataset(Dataset):
    """
    PyTorch Dataset for the Denoising Task.
    Handles loading of patch pairs and applying consistent augmentations.
    """

    def __init__(self, noisy_patches, clean_patches, augment=False):
        """
        Args:
            noisy_patches (np.ndarray): Array of shape (N, 1, H, W) or (N, H, W).
            clean_patches (np.ndarray): Array of shape (N, 1, H, W) or (N, H, W).
            augment (bool): Whether to apply random geometric augmentations.
        """
        self.noisy_patches = torch.from_numpy(noisy_patches).float()
        self.clean_patches = torch.from_numpy(clean_patches).float()
        self.augment = augment

        # Ensure channel dimension exists
        if self.noisy_patches.ndim == 3:
            self.noisy_patches = self.noisy_patches.unsqueeze(1)
        if self.clean_patches.ndim == 3:
            self.clean_patches = self.clean_patches.unsqueeze(1)

    def __len__(self):
        return len(self.noisy_patches)

    def __getitem__(self, idx):
        noisy = self.noisy_patches[idx]
        clean = self.clean_patches[idx]

        if self.augment:
            # Apply random horizontal flip
            if torch.rand(1) < 0.5:
                noisy = torch.flip(noisy, [2])
                clean = torch.flip(clean, [2])

            # Apply random vertical flip
            if torch.rand(1) < 0.5:
                noisy = torch.flip(noisy, [1])
                clean = torch.flip(clean, [1])

            # Apply random 90-degree rotation (0, 1, 2, or 3 times)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                noisy = torch.rot90(noisy, k, [1, 2])
                clean = torch.rot90(clean, k, [1, 2])

        return noisy, clean


def extract_patches(metadata_df, stride, patch_size=Config.PATCH_SIZE):
    """
    Extracts patches from images listed in the metadata DataFrame.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'input_path' and 'target_path'.
        stride (int): Stride for the sliding window.
        patch_size (int): Size of the square patch.

    Returns:
        tuple: (noisy_patches, clean_patches) as numpy arrays of shape (N, 1, H, W).
    """
    noisy_patches = []
    clean_patches = []

    # Limit for debugging if enabled
    if Config.DEBUG:
        metadata_df = metadata_df.head(5)

    for _, row in metadata_df.iterrows():
        input_path = os.path.join(Config.INPUT_DIR, row["input_path"])
        target_path = os.path.join(Config.INPUT_DIR, row["target_path"])

        # Load images in grayscale
        img_noisy = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        img_clean = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        if img_noisy is None or img_clean is None:
            continue

        # Normalize to [0, 1]
        img_noisy = img_noisy.astype(np.float32) / 255.0
        img_clean = img_clean.astype(np.float32) / 255.0

        h, w = img_noisy.shape

        # Extract patches
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch_n = img_noisy[y : y + patch_size, x : x + patch_size]
                patch_c = img_clean[y : y + patch_size, x : x + patch_size]

                noisy_patches.append(patch_n)
                clean_patches.append(patch_c)

                if Config.DEBUG and len(noisy_patches) >= Config.DEBUG_SAMPLE_SIZE:
                    break
            if Config.DEBUG and len(noisy_patches) >= Config.DEBUG_SAMPLE_SIZE:
                break

    # Convert to numpy arrays and add channel dimension
    noisy_patches = np.array(noisy_patches)[:, np.newaxis, :, :]
    clean_patches = np.array(clean_patches)[:, np.newaxis, :, :]

    return noisy_patches, clean_patches


def load_dataset_patches(mode, load_cached_data=True):
    """
    Loads patches for the specified mode, using caching to save time.

    Args:
        mode (str): One of 'train_sparse', 'train_dense', 'val'.
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        tuple: (noisy_patches, clean_patches) numpy arrays.
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    patches_path = os.path.join(cache_dir, f"{mode}_patches.npy")
    targets_path = os.path.join(cache_dir, f"{mode}_targets.npy")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(patches_path)
        and os.path.exists(targets_path)
    ):
        print(f"Loading {mode} data from cache...")
        try:
            patches = np.load(patches_path)
            targets = np.load(targets_path)
            print(f"Loaded {len(patches)} patches for {mode}.")
            return patches, targets
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Generating {mode} data from scratch...")

    if mode == "train_sparse":
        df = pd.read_csv(Config.TRAIN_CSV)
        stride = Config.STRIDE_SPARSE
    elif mode == "train_dense":
        df = pd.read_csv(Config.TRAIN_CSV)
        stride = Config.STRIDE_DENSE
    elif mode == "val":
        df = pd.read_csv(Config.VAL_CSV)
        # Use sparse stride for validation to keep evaluation reasonably fast but representative
        stride = Config.STRIDE_SPARSE
    else:
        raise ValueError(f"Unknown mode: {mode}")

    patches, targets = extract_patches(df, stride)

    # 3. Save to cache
    print(f"Saving {mode} data to cache...")
    np.save(patches_path, patches)
    np.save(targets_path, targets)

    print(f"Generated {len(patches)} patches for {mode}.")
    return patches, targets
