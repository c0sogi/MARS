import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    PATCH_SIZE,
    STRIDE_SPARSE,
    STRIDE_DENSE,
    SEED,
)


class DenoisingDataset(Dataset):
    """
    PyTorch Dataset for Denoising Task.
    Handles on-the-fly geometric augmentations for training.
    """

    def __init__(self, inputs, targets, augment=False):
        """
        Args:
            inputs (np.ndarray): Input patches (N, 1, H, W).
            targets (np.ndarray): Target patches (N, 1, H, W).
            augment (bool): Whether to apply geometric augmentations.
        """
        self.inputs = inputs
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Retrieve patch: (1, H, W)
        x = self.inputs[idx]
        y = self.targets[idx]

        if self.augment:
            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)

            # Random Vertical Flip
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)

            # Random Rotation (0, 90, 180, 270 degrees)
            k = np.random.randint(0, 4)
            if k > 0:
                x = np.rot90(x, k, axes=(1, 2))
                y = np.rot90(y, k, axes=(1, 2))

        # Ensure arrays are contiguous in memory after numpy strides
        x = np.ascontiguousarray(x).astype(np.float32)
        y = np.ascontiguousarray(y).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


def load_images(metadata_path, limit=None):
    """
    Loads paired images from disk based on metadata.

    Args:
        metadata_path (str): Path to the metadata CSV.
        limit (int, optional): Max number of images to load (for debugging).

    Returns:
        tuple: (inputs_list, targets_list) containing normalized float32 images.
    """
    df = pd.read_csv(metadata_path)
    if limit is not None:
        df = df.head(limit)

    inputs = []
    targets = []

    for _, row in df.iterrows():
        # Construct paths
        in_path = os.path.join(INPUT_DIR, row["input_path"])

        # Load Input
        img_in = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if img_in is None:
            continue

        # Load Target
        if "target_path" in row and pd.notna(row["target_path"]):
            tar_path = os.path.join(INPUT_DIR, row["target_path"])
            img_tar = cv2.imread(tar_path, cv2.IMREAD_GRAYSCALE)

            if img_tar is None:
                continue

            # Normalize to [0, 1]
            inputs.append(img_in.astype(np.float32) / 255.0)
            targets.append(img_tar.astype(np.float32) / 255.0)

    return inputs, targets


def extract_patches(images, patch_size, stride):
    """
    Extracts patches from a list of images.

    Args:
        images (list): List of 2D numpy arrays (H, W).
        patch_size (int): Size of the square patch.
        stride (int): Stride for extraction.

    Returns:
        np.ndarray: 4D array of shape (N, 1, patch_size, patch_size).
    """
    patches = []
    for img in images:
        h, w = img.shape
        # Slide window
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = img[y : y + patch_size, x : x + patch_size]
                patches.append(patch)

    if not patches:
        return np.empty((0, 1, patch_size, patch_size), dtype=np.float32)

    # Convert to numpy array and add channel dimension
    patches_np = np.array(patches, dtype=np.float32)
    patches_np = np.expand_dims(patches_np, axis=1)  # (N, 1, H, W)
    return patches_np


def get_processed_data(
    mode="train", stride_type="sparse", load_cached_data=True, limit=None
):
    """
    Retrieves processed patches, using caching to save time.

    Args:
        mode (str): 'train' or 'val'.
        stride_type (str): 'sparse' or 'dense'.
        load_cached_data (bool): If True, attempts to load from .npy cache.
        limit (int, optional): Limit number of source images.

    Returns:
        tuple: (patches, targets) as numpy arrays.
    """
    # Configure based on mode
    if mode == "train":
        metadata_path = TRAIN_METADATA_PATH
        stride = STRIDE_SPARSE if stride_type == "sparse" else STRIDE_DENSE
    elif mode == "val":
        metadata_path = VAL_METADATA_PATH
        # Always use sparse stride for validation to keep evaluation fast
        stride = STRIDE_SPARSE
    else:
        raise ValueError("Mode must be 'train' or 'val'")

    # Construct cache filenames
    suffix = f"{mode}_{stride_type}"
    if limit is not None:
        suffix += f"_limit{limit}"

    cache_patches_path = os.path.join(WORKING_DIR, f"patches_{suffix}.npy")
    cache_targets_path = os.path.join(WORKING_DIR, f"targets_{suffix}.npy")

    # 1. Attempt to load from cache
    if load_cached_data:
        if os.path.exists(cache_patches_path) and os.path.exists(cache_targets_path):
            try:
                patches = np.load(cache_patches_path)
                targets = np.load(cache_targets_path)
                return patches, targets
            except Exception:
                # If load fails, proceed to re-process
                pass

    # 2. Process from scratch
    # Ensure output directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Load images
    input_imgs, target_imgs = load_images(metadata_path, limit=limit)

    # Extract patches
    patches = extract_patches(input_imgs, PATCH_SIZE, stride)
    targets = extract_patches(target_imgs, PATCH_SIZE, stride)

    # Save to cache
    np.save(cache_patches_path, patches)
    np.save(cache_targets_path, targets)

    return patches, targets


def load_test_data():
    """
    Loads full test images for inference.

    Returns:
        tuple: (image_ids, images_list)
               image_ids is a list of filename strings.
               images_list is a list of normalized float32 numpy arrays.
    """
    df = pd.read_csv(TEST_METADATA_PATH)
    ids = []
    images = []

    for _, row in df.iterrows():
        img_id = row["image_id"]
        path = os.path.join(INPUT_DIR, row["input_path"])

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = img.astype(np.float32) / 255.0
            ids.append(img_id)
            images.append(img)

    return ids, images
