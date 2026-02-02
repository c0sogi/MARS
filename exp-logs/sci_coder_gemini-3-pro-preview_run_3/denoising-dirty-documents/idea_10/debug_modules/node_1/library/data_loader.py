import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library import config, utils


def extract_patches(image, patch_size, stride):
    """
    Extracts patches from a single image using a sliding window.

    Args:
        image (np.ndarray): Input image (H, W).
        patch_size (int): Size of the square patch.
        stride (int): Stride for the sliding window.

    Returns:
        np.ndarray: Array of patches (N, patch_size, patch_size).
    """
    h, w = image.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(patch)
    return np.array(patches)


def prepare_data(load_cached_data=True, max_samples=None):
    """
    Loads data from metadata, extracts patches, and caches them.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        max_samples (int, optional): Limit the number of samples for debugging.

    Returns:
        tuple: (train_patches, train_targets, val_patches, val_targets)
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Check if cached files exist
    files_exist = (
        os.path.exists(config.TRAIN_PATCHES_PATH)
        and os.path.exists(config.TRAIN_TARGETS_PATH)
        and os.path.exists(config.VAL_PATCHES_PATH)
        and os.path.exists(config.VAL_TARGETS_PATH)
    )

    if load_cached_data and files_exist:
        print("Loading cached data...")
        try:
            train_patches = np.load(config.TRAIN_PATCHES_PATH)
            train_targets = np.load(config.TRAIN_TARGETS_PATH)
            val_patches = np.load(config.VAL_PATCHES_PATH)
            val_targets = np.load(config.VAL_TARGETS_PATH)

            # If max_samples is set, slice the cached data
            if max_samples is not None:
                # We assume the cache is shuffled or order doesn't matter for slicing
                # But typically we slice by original images. Here we slice by patches.
                # Given the volume, we just take the first N patches.
                limit = max_samples * 100  # Rough estimate: 100 patches per image
                train_patches = train_patches[:limit]
                train_targets = train_targets[:limit]
                val_patches = val_patches[:limit]
                val_targets = val_targets[:limit]

            return train_patches, train_targets, val_patches, val_targets
        except Exception as e:
            print(f"Failed to load cached data: {e}. Recomputing from scratch...")
            # Fall through to computation

    print("Processing data from scratch...")

    # Load Metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)

    # Apply max_samples to metadata (images) if provided
    if max_samples is not None:
        df_train = df_train.iloc[:max_samples]
        df_val = df_val.iloc[:max_samples]

    def process_split(df):
        patches_list = []
        targets_list = []

        for _, row in df.iterrows():
            input_path = os.path.join(config.INPUT_DIR, row["input_path"])
            target_path = os.path.join(config.INPUT_DIR, row["target_path"])

            # Read images
            img_in = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            img_tar = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

            if img_in is None or img_tar is None:
                continue

            # Normalize [0, 1]
            img_in = utils.normalize_image(img_in)
            img_tar = utils.normalize_image(img_tar)

            # Extract patches
            p_in = extract_patches(img_in, config.PATCH_SIZE, config.STRIDE)
            p_tar = extract_patches(img_tar, config.PATCH_SIZE, config.STRIDE)

            patches_list.append(p_in)
            targets_list.append(p_tar)

        if not patches_list:
            return np.empty((0, config.PATCH_SIZE, config.PATCH_SIZE)), np.empty(
                (0, config.PATCH_SIZE, config.PATCH_SIZE)
            )

        return np.concatenate(patches_list), np.concatenate(targets_list)

    train_patches, train_targets = process_split(df_train)
    val_patches, val_targets = process_split(df_val)

    # Cache data ONLY if we processed the full dataset (max_samples is None)
    # This prevents overwriting the full cache with a partial debug set.
    if max_samples is None:
        np.save(config.TRAIN_PATCHES_PATH, train_patches)
        np.save(config.TRAIN_TARGETS_PATH, train_targets)
        np.save(config.VAL_PATCHES_PATH, val_patches)
        np.save(config.VAL_TARGETS_PATH, val_targets)
        print("Data cached successfully.")
    else:
        print("Skipping cache save (max_samples set).")

    return train_patches, train_targets, val_patches, val_targets


class DenoisingDataset(Dataset):
    def __init__(self, patches, targets, augment=False):
        """
        PyTorch Dataset for denoising.

        Args:
            patches (np.ndarray): Input noisy patches.
            targets (np.ndarray): Clean target patches.
            augment (bool): Whether to apply data augmentation.
        """
        # Convert to tensor and add channel dimension (N, 1, H, W)
        self.patches = torch.from_numpy(patches).float().unsqueeze(1)
        self.targets = torch.from_numpy(targets).float().unsqueeze(1)
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = self.patches[idx]
        y = self.targets[idx]

        if self.augment:
            # Random flip Horizontal
            if torch.rand(1) < 0.5:
                x = torch.flip(x, [2])
                y = torch.flip(y, [2])
            # Random flip Vertical
            if torch.rand(1) < 0.5:
                x = torch.flip(x, [1])
                y = torch.flip(y, [1])
            # Random Rotate 90 (0, 90, 180, 270 degrees)
            k = torch.randint(0, 4, (1,)).item()
            x = torch.rot90(x, k, [1, 2])
            y = torch.rot90(y, k, [1, 2])

        # Target for network is NOISE (Input - Clean)
        # We predict the residual noise to subtract from the input
        noise = x - y

        return x, noise
