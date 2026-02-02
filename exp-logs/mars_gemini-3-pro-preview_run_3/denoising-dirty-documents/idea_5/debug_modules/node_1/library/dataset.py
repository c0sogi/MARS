import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_image


def extract_dense_patches(image, patch_size, stride):
    """
    Extracts patches from an image with a specified stride.

    Args:
        image (np.ndarray): Input image array of shape (H, W).
        patch_size (int): The height and width of the patch.
        stride (int): The step size for the sliding window.

    Returns:
        np.ndarray: Array of extracted patches of shape (N, patch_size, patch_size).
    """
    h, w = image.shape
    patches = []

    # Calculate starting indices for the sliding window
    # Ensure we stop such that the patch fits entirely within the image
    y_range = range(0, h - patch_size + 1, stride)
    x_range = range(0, w - patch_size + 1, stride)

    for y in y_range:
        for x in x_range:
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(patch)

    if not patches:
        return np.empty((0, patch_size, patch_size), dtype=image.dtype)

    return np.array(patches)


class DenoisingDataset(Dataset):
    """
    PyTorch Dataset for the Denoising Task.
    Handles on-the-fly augmentation and tensor conversion.
    """

    def __init__(self, patches, targets=None, augment=False):
        """
        Args:
            patches (np.ndarray): Input noisy patches (N, H, W).
            targets (np.ndarray, optional): Target noise patches (N, H, W).
            augment (bool): Whether to apply random geometric augmentations.
        """
        self.patches = patches
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        target = self.targets[idx] if self.targets is not None else None

        if self.augment:
            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=1)
                if target is not None:
                    target = np.flip(target, axis=1)

            # Random Vertical Flip
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=0)
                if target is not None:
                    target = np.flip(target, axis=0)

            # Random 90-degree Rotation
            k = np.random.randint(0, 4)
            if k > 0:
                patch = np.rot90(patch, k=k)
                if target is not None:
                    target = np.rot90(target, k=k)

        # Convert to Tensor and add Channel dimension: (H, W) -> (1, H, W)
        # Using .copy() to handle negative strides from flip/rot which torch doesn't support directly
        patch_tensor = torch.from_numpy(patch.copy()).float().unsqueeze(0)

        if target is not None:
            target_tensor = torch.from_numpy(target.copy()).float().unsqueeze(0)
            return patch_tensor, target_tensor

        return patch_tensor


def _process_split(metadata_path, stride):
    """
    Helper function to process a dataset split from metadata.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    all_patches = []
    all_targets = []

    for _, row in df.iterrows():
        input_rel_path = row["input_path"]
        target_rel_path = row["target_path"]

        input_path = os.path.join(Config.INPUT_DIR, input_rel_path)
        target_path = os.path.join(Config.INPUT_DIR, target_rel_path)

        # Load images (normalized 0-1 grayscale)
        img_in = load_image(input_path)
        img_tar = load_image(target_path)

        # Calculate Noise Residual (Input - Clean)
        # The model predicts this residual.
        noise_map = img_in - img_tar

        # Extract Patches
        patches_in = extract_dense_patches(img_in, Config.PATCH_SIZE, stride)
        patches_noise = extract_dense_patches(noise_map, Config.PATCH_SIZE, stride)

        all_patches.append(patches_in)
        all_targets.append(patches_noise)

    if not all_patches:
        empty = np.empty((0, Config.PATCH_SIZE, Config.PATCH_SIZE), dtype=np.float32)
        return empty, empty

    return np.concatenate(all_patches, axis=0), np.concatenate(all_targets, axis=0)


def prepare_datasets(load_cached_data=True):
    """
    Prepares the training and validation datasets.
    Implements caching to avoid re-processing images.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    train_cache = Config.TRAIN_PATCHES_CACHE
    val_cache = Config.VAL_PATCHES_CACHE

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(train_cache) and os.path.exists(val_cache):
        print(f"Loading cached patches from {Config.WORKING_DIR}...")
        try:
            train_data = np.load(train_cache, allow_pickle=True).item()
            val_data = np.load(val_cache, allow_pickle=True).item()

            train_dataset = DenoisingDataset(
                train_data["patches"], train_data["targets"], augment=True
            )
            val_dataset = DenoisingDataset(
                val_data["patches"], val_data["targets"], augment=False
            )
            return train_dataset, val_dataset
        except Exception as e:
            print(f"Error loading cache: {e}. Regenerating data...")

    # 2. Process from Scratch
    print("Generating patches from scratch...")

    # Process Train: Use dense stride for high data density
    print(f"Processing Training Set (Stride={Config.STRIDE})...")
    train_patches, train_targets = _process_split(Config.TRAIN_METADATA, Config.STRIDE)

    # Process Validation: Use patch_size stride (non-overlapping) for faster evaluation
    print(f"Processing Validation Set (Stride={Config.PATCH_SIZE})...")
    val_patches, val_targets = _process_split(Config.VAL_METADATA, Config.PATCH_SIZE)

    # 3. Save to Cache
    print(f"Saving data to {Config.WORKING_DIR}...")
    np.save(train_cache, {"patches": train_patches, "targets": train_targets})
    np.save(val_cache, {"patches": val_patches, "targets": val_targets})

    # 4. Create Datasets
    # Enable augmentation only for training
    train_dataset = DenoisingDataset(train_patches, train_targets, augment=True)
    val_dataset = DenoisingDataset(val_patches, val_targets, augment=False)

    return train_dataset, val_dataset
