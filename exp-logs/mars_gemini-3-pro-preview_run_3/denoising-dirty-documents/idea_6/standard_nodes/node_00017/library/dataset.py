import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library import config
from library import utils


class DenoisingDataset(Dataset):
    """
    PyTorch Dataset for the Denoising Task.
    Serves pairs of (noisy_patch, noise_residual_target).
    """

    def __init__(self, patches, targets, augment=False):
        """
        Args:
            patches (np.ndarray): Array of noisy image patches (N, 1, H, W).
            targets (np.ndarray): Array of noise residual targets (N, 1, H, W).
            augment (bool): Whether to apply random augmentations (flips, rotations).
        """
        # Convert to torch tensors immediately to avoid overhead during training loop
        # Assuming patches are already float32 and normalized [0, 1]
        self.patches = torch.from_numpy(patches)
        self.targets = torch.from_numpy(targets)
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = self.patches[idx]
        y = self.targets[idx]

        if self.augment:
            # Apply random augmentations
            # 1. Random Horizontal Flip
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])

            # 2. Random Vertical Flip
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])

            # 3. Random 90-degree Rotation
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                x = torch.rot90(x, k, dims=[1, 2])
                y = torch.rot90(y, k, dims=[1, 2])

        return x, y


def _extract_dense_patches(image, patch_size, stride):
    """
    Extracts overlapping patches from an image.

    Args:
        image (np.ndarray): Input image of shape (H, W).
        patch_size (int): Height/Width of the patch.
        stride (int): Step size for extraction.

    Returns:
        np.ndarray: Array of patches with shape (N, 1, patch_size, patch_size).
    """
    h, w = image.shape
    patches = []

    # Slide window
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(patch)

    if not patches:
        return np.empty((0, 1, patch_size, patch_size), dtype=np.float32)

    # Stack and add channel dimension
    return np.array(patches, dtype=np.float32)[:, np.newaxis, :, :]


def _process_split(metadata_path, patch_size, stride, desc="Data"):
    """
    Loads images defined in metadata, calculates residuals, and extracts patches.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    all_patches = []
    all_targets = []

    print(f"Processing {desc} from {metadata_path}...")

    for _, row in df.iterrows():
        input_rel_path = row["input_path"]
        target_rel_path = row["target_path"]

        # Construct full paths
        input_path = os.path.join(config.INPUT_DIR, input_rel_path)
        target_path = os.path.join(config.INPUT_DIR, target_rel_path)

        # Load images (normalized [0, 1])
        img_noisy = utils.load_grayscale_image(input_path)
        img_clean = utils.load_grayscale_image(target_path)

        # Calculate Noise Residual: Target = Input - Clean
        # The model will learn to predict this noise
        noise_residual = img_noisy - img_clean

        # Extract patches
        img_patches = _extract_dense_patches(img_noisy, patch_size, stride)
        res_patches = _extract_dense_patches(noise_residual, patch_size, stride)

        all_patches.append(img_patches)
        all_targets.append(res_patches)

    # Concatenate all patches from all images
    if all_patches:
        X = np.concatenate(all_patches, axis=0)
        y = np.concatenate(all_targets, axis=0)
    else:
        X = np.empty((0, 1, patch_size, patch_size), dtype=np.float32)
        y = np.empty((0, 1, patch_size, patch_size), dtype=np.float32)

    return X, y


def prepare_datasets(load_cached_data=True):
    """
    Prepares the training and validation datasets.
    Handles caching of processed patches to disk.

    Args:
        load_cached_data (bool): If True, attempts to load from .npy files.

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_x_path = config.TRAIN_PATCHES_CACHE
    train_y_path = config.TRAIN_TARGETS_CACHE
    val_x_path = config.VAL_PATCHES_CACHE
    val_y_path = config.VAL_TARGETS_CACHE

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_x_path)
        and os.path.exists(train_y_path)
        and os.path.exists(val_x_path)
        and os.path.exists(val_y_path)
    )

    if load_cached_data and cache_exists:
        print("Loading datasets from cache...")
        X_train = np.load(train_x_path)
        y_train = np.load(train_y_path)
        X_val = np.load(val_x_path)
        y_val = np.load(val_y_path)
    else:
        print("Generating datasets from scratch...")

        # Process Training Data
        # High density: small stride
        X_train, y_train = _process_split(
            config.TRAIN_METADATA_PATH,
            config.PATCH_SIZE,
            config.STRIDE,
            desc="Training",
        )

        # Process Validation Data
        # We also patch validation data to compute validation loss (MSE)
        # We use the same stride/patch size for consistency in loss calculation,
        # though strictly speaking validation could be non-overlapping.
        # Using same settings ensures distribution match.
        X_val, y_val = _process_split(
            config.VAL_METADATA_PATH,
            config.PATCH_SIZE,
            config.STRIDE,
            desc="Validation",
        )

        # Save to cache
        print(f"Saving cache to {config.WORKING_DIR}...")
        np.save(train_x_path, X_train)
        np.save(train_y_path, y_train)
        np.save(val_x_path, X_val)
        np.save(val_y_path, y_val)

    print(f"Train Data Shape: {X_train.shape}")
    print(f"Val Data Shape:   {X_val.shape}")

    # Create Dataset objects
    # Enable augmentation only for training
    train_dataset = DenoisingDataset(X_train, y_train, augment=True)
    val_dataset = DenoisingDataset(X_val, y_val, augment=False)

    return train_dataset, val_dataset
