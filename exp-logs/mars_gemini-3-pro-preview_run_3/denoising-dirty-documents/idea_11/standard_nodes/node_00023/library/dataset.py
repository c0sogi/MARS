import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def load_image(path):
    """
    Load an image from disk, convert to grayscale, and normalize to [0, 1].
    """
    # Load as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    return img


def extract_patches(
    metadata_path,
    stride,
    patch_size,
    cache_patches_path,
    cache_targets_path,
    load_cached_data=True,
    is_test=False,
):
    """
    Extract patches from images listed in the metadata CSV.
    Implements caching logic using .npy files.

    Args:
        metadata_path: Path to the CSV file containing image paths.
        stride: Stride for patch extraction.
        patch_size: Size of the square patch.
        cache_patches_path: Path to save/load input patches.
        cache_targets_path: Path to save/load target patches (None for test).
        load_cached_data: Whether to attempt loading from cache.
        is_test: If True, ignores target paths.

    Returns:
        patches: Numpy array of shape (N, patch_size, patch_size)
        targets: Numpy array of shape (N, patch_size, patch_size) or None
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_patches_path), exist_ok=True)
    if cache_targets_path:
        os.makedirs(os.path.dirname(cache_targets_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_patches_path):
            if is_test or (cache_targets_path and os.path.exists(cache_targets_path)):
                print(f"Loading cached patches from {cache_patches_path}...")
                patches = np.load(cache_patches_path)
                targets = np.load(cache_targets_path) if not is_test else None
                return patches, targets

    # 2. Process from scratch
    print(f"Extracting patches from {metadata_path} with stride {stride}...")
    df = pd.read_csv(metadata_path)

    patch_list = []
    target_list = []

    for _, row in df.iterrows():
        # Construct full paths
        input_path = os.path.join(Config.INPUT_DIR, row["input_path"])

        # Load Input Image
        img_in = load_image(input_path)
        h, w = img_in.shape

        # Load Target Image if not test
        img_tar = None
        if not is_test:
            target_path = os.path.join(Config.INPUT_DIR, row["target_path"])
            img_tar = load_image(target_path)

            # Sanity check dimensions
            if img_in.shape != img_tar.shape:
                raise ValueError(
                    f"Shape mismatch for {row['image_id']}: {img_in.shape} vs {img_tar.shape}"
                )

        # Extract Patches
        # We iterate strictly within bounds
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch_in = img_in[y : y + patch_size, x : x + patch_size]
                patch_list.append(patch_in)

                if not is_test:
                    patch_tar = img_tar[y : y + patch_size, x : x + patch_size]
                    target_list.append(patch_tar)

    # Convert to numpy arrays
    patches = np.array(patch_list, dtype=np.float32)
    targets = np.array(target_list, dtype=np.float32) if not is_test else None

    # 3. Save to cache
    print(f"Saving {len(patches)} patches to cache...")
    np.save(cache_patches_path, patches)
    if not is_test:
        np.save(cache_targets_path, targets)

    return patches, targets


class DenoisingDataset(Dataset):
    """
    PyTorch Dataset for Denoising.
    Handles augmentation and tensor conversion.
    """

    def __init__(self, patches, targets=None, augment=False):
        """
        Args:
            patches: Numpy array (N, H, W)
            targets: Numpy array (N, H, W) or None
            augment: Boolean, whether to apply geometric augmentations
        """
        self.patches = patches
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Get data
        input_patch = self.patches[idx]
        target_patch = self.targets[idx] if self.targets is not None else None

        # Apply Augmentation (only for training)
        if self.augment:
            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                input_patch = np.fliplr(input_patch)
                if target_patch is not None:
                    target_patch = np.fliplr(target_patch)

            # Random Vertical Flip
            if np.random.rand() > 0.5:
                input_patch = np.flipud(input_patch)
                if target_patch is not None:
                    target_patch = np.flipud(target_patch)

            # Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                input_patch = np.rot90(input_patch, k)
                if target_patch is not None:
                    target_patch = np.rot90(target_patch, k)

        # Convert to Tensor and add Channel Dimension (C, H, W)
        # Input is (H, W) -> (1, H, W)
        input_tensor = torch.from_numpy(input_patch.copy()).unsqueeze(0)

        if target_patch is not None:
            target_tensor = torch.from_numpy(target_patch.copy()).unsqueeze(0)
            # Calculate noise residual for training target if needed,
            # but usually the dataset returns clean and model/loss handles residual calculation.
            # However, the task description says: "predict the noise residual".
            # It is often cleaner to return Input and Clean Target, and compute Residual = Input - Clean inside the training loop.
            # I will return (Input, Clean Target) here.
            return input_tensor, target_tensor
        else:
            return input_tensor
