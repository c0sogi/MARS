import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def extract_patches(image, patch_size, stride):
    """
    Extracts patches from a single image with the specified stride.
    Returns a numpy array of shape (N, H, W).
    """
    h, w = image.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(patch)

    if not patches:
        return np.empty((0, patch_size, patch_size), dtype=image.dtype)

    return np.array(patches)


def prepare_data(metadata_path, cache_path, load_cached_data=True):
    """
    Loads images based on metadata, extracts patches, and caches the result.
    If cached data exists and load_cached_data is True, loads from disk.

    Args:
        metadata_path (str): Path to the CSV file containing image metadata.
        cache_path (str): Path where the processed numpy array should be saved/loaded.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Array of shape (N, 2, H, W) containing patch pairs.
    """
    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            print(f"Loaded cached data from {cache_path}")
            return data
        except Exception as e:
            print(f"Error loading cache from {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Apply debugging limits if specified in Config
    # We check the filename to distinguish between train and val sets
    filename = os.path.basename(metadata_path)
    if Config.MAX_TRAIN_IMAGES is not None and "train" in filename:
        df = df.head(Config.MAX_TRAIN_IMAGES)
    if Config.MAX_VAL_IMAGES is not None and "val" in filename:
        df = df.head(Config.MAX_VAL_IMAGES)

    patch_pairs = []

    for _, row in df.iterrows():
        input_full_path = os.path.join(Config.INPUT_DIR, row["input_path"])
        target_full_path = os.path.join(Config.INPUT_DIR, row["target_path"])

        # Load as Grayscale
        img_in = cv2.imread(input_full_path, cv2.IMREAD_GRAYSCALE)
        img_tar = cv2.imread(target_full_path, cv2.IMREAD_GRAYSCALE)

        if img_in is None or img_tar is None:
            continue

        # Normalize to [0, 1]
        img_in = img_in.astype(np.float32) / 255.0
        img_tar = img_tar.astype(np.float32) / 255.0

        # Extract patches
        p_in = extract_patches(img_in, Config.PATCH_SIZE, Config.STRIDE)
        p_tar = extract_patches(img_tar, Config.PATCH_SIZE, Config.STRIDE)

        # Store as pairs if patches exist and match in count
        if len(p_in) > 0 and len(p_in) == len(p_tar):
            # Stack to shape (N, 2, H, W)
            pairs = np.stack([p_in, p_tar], axis=1)
            patch_pairs.append(pairs)

    if len(patch_pairs) > 0:
        all_data = np.concatenate(patch_pairs, axis=0)
    else:
        all_data = np.empty(
            (0, 2, Config.PATCH_SIZE, Config.PATCH_SIZE), dtype=np.float32
        )

    # 3. Save to cache
    try:
        np.save(cache_path, all_data)
        print(f"Saved cache to {cache_path}")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return all_data


class DenoisingDataset(Dataset):
    def __init__(self, data, augment=False):
        """
        Args:
            data (np.ndarray): Numpy array of shape (N, 2, H, W) where 2 corresponds to [noisy, clean].
            augment (bool): Whether to apply geometric augmentations.
        """
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy, clean = self.data[idx]

        if self.augment:
            # Random Horizontal Flip
            if np.random.rand() < 0.5:
                noisy = np.flip(noisy, axis=1)
                clean = np.flip(clean, axis=1)
            # Random Vertical Flip
            if np.random.rand() < 0.5:
                noisy = np.flip(noisy, axis=0)
                clean = np.flip(clean, axis=0)
            # Random Rotation (0, 90, 180, 270 degrees)
            k = np.random.randint(0, 4)
            if k > 0:
                noisy = np.rot90(noisy, k)
                clean = np.rot90(clean, k)

        # Ensure contiguous arrays after flipping/rotation
        noisy = np.ascontiguousarray(noisy)
        clean = np.ascontiguousarray(clean)

        # Convert to Tensor and add channel dimension: (1, H, W)
        noisy_t = torch.from_numpy(noisy).unsqueeze(0)
        clean_t = torch.from_numpy(clean).unsqueeze(0)

        return noisy_t, clean_t
