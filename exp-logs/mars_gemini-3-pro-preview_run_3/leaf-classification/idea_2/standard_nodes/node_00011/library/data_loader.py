import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class LeafImageDataset(Dataset):
    """
    PyTorch Dataset for loading binary leaf images.

    Performs the following preprocessing:
    1. Loads image (grayscale).
    2. Resizes to Config.IMG_SIZE.
    3. Converts to float [0, 1].
    4. Replicates the single channel 3 times to create pseudo-RGB.
    5. Normalizes using ImageNet mean and std.
    """

    def __init__(self, metadata_df, transform=None):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'file_path', 'id', and optionally 'species'.
            transform (callable, optional): Optional transform to be applied on a sample (e.g. augmentations).
        """
        self.metadata_df = metadata_df
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

        # Prepare normalization constants (C, 1, 1) for broadcasting or (1, 1, C) depending on format
        # Here we process in HWC format (numpy) before converting to CHW (tensor)
        self.mean = np.array(Config.IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(Config.IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]

        # Construct full file path
        rel_path = row["file_path"]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load image
        # The images are binary black leaves on white background.
        # We load as grayscale.
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Safety fallback for missing files
            img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # Resize to target dimensions
        img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))

        # Convert to float32 and scale to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Replicate channels to create pseudo-RGB (H, W, 3)
        img = np.stack([img] * 3, axis=-1)

        # Normalize with ImageNet stats
        img = (img - self.mean) / self.std

        # Transpose to PyTorch format (C, H, W)
        img = img.transpose(2, 0, 1)

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()

        # Retrieve ID and Target
        image_id = row["id"]
        target = row["species"] if "species" in row else -1

        # Apply external transforms if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, target, image_id


def load_tabular_data(split="train", load_cached_data=True, limit=None):
    """
    Loads tabular features (margin, shape, texture) and targets from metadata CSVs.
    Implements deterministic caching using .npy files.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.
        limit (int, optional): If provided, returns only the first 'limit' samples.

    Returns:
        tuple: (X, y, ids)
            X (np.ndarray): Feature matrix of shape (N, 192).
            y (np.ndarray or None): Target array of shape (N,). None for 'test' split.
            ids (np.ndarray): ID array of shape (N,).
    """
    # Ensure the cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache filenames
    cache_X_path = os.path.join(Config.CACHE_DIR, f"{split}_tabular_X.npy")
    cache_y_path = os.path.join(Config.CACHE_DIR, f"{split}_tabular_y.npy")
    cache_ids_path = os.path.join(Config.CACHE_DIR, f"{split}_tabular_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        # Check if X and ids exist. For y, check only if not test split.
        has_X_ids = os.path.exists(cache_X_path) and os.path.exists(cache_ids_path)
        has_y = os.path.exists(cache_y_path) or split == "test"

        if has_X_ids and has_y:
            print(f"Loading {split} tabular data from cache: {Config.CACHE_DIR}")
            X = np.load(cache_X_path)
            ids = np.load(cache_ids_path)
            y = np.load(cache_y_path, allow_pickle=True) if split != "test" else None

            # Apply limit if requested
            if limit is not None:
                return X[:limit], (y[:limit] if y is not None else None), ids[:limit]
            return X, y, ids

    # 2. If cache miss or forced reload, process from scratch
    print(f"Processing {split} tabular data from source metadata...")

    # Determine source CSV path
    if split == "train":
        source_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        source_path = Config.VAL_METADATA_PATH
    elif split == "test":
        source_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # Load CSV
    df = pd.read_csv(source_path)

    # Extract IDs
    ids = df["id"].values

    # Extract Features
    # Identify columns for margin, shape, and texture
    # We assume the CSV columns are consistent with the dataset description
    feature_cols = [
        c
        for c in df.columns
        if c.startswith("margin") or c.startswith("shape") or c.startswith("texture")
    ]
    X = df[feature_cols].values.astype(np.float32)

    # Extract Targets
    if split != "test":
        y = df["species"].values
    else:
        y = None

    # Save to cache
    np.save(cache_X_path, X)
    np.save(cache_ids_path, ids)
    if y is not None:
        np.save(cache_y_path, y)

    # Apply limit if requested
    if limit is not None:
        return X[:limit], (y[:limit] if y is not None else None), ids[:limit]

    return X, y, ids
