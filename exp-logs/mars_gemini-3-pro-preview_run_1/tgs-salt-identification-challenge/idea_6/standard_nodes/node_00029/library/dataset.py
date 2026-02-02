import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.utils import pad_image_128

# Constants for Depth Normalization
# Data range is approx 51-959. We use 0-1000 to keep it in a safe 0-1 range.
DEPTH_MIN = 0.0
DEPTH_MAX = 1000.0


class SaltDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        root_dir="./input",
        mode="train",
        transform=False,
        load_cached_data=True,
        cache_dir="./working/idea_6",
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            root_dir (str): Root directory of the dataset.
            mode (str): 'train', 'val', or 'test'.
            transform (bool): If True, applies augmentation (Horizontal Flip) for training.
            load_cached_data (bool): Whether to load data from cache if available.
            cache_dir (str): Directory to store/load cached numpy arrays.
        """
        self.mode = mode
        self.transform = transform
        self.root_dir = root_dir
        self.metadata_path = metadata_path

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Define cache filenames based on mode
        # We use the mode string to differentiate caches (train vs val vs test)
        cache_prefix = os.path.join(cache_dir, f"{mode}")
        self.images_path = f"{cache_prefix}_images.npy"
        self.masks_path = f"{cache_prefix}_masks.npy"
        self.depths_path = f"{cache_prefix}_depths.npy"
        self.ids_path = f"{cache_prefix}_ids.npy"

        # Load Data
        self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        """
        Loads data from cache or processes it from scratch.
        """
        # Check if all required cache files exist
        cache_exists = (
            os.path.exists(self.images_path)
            and os.path.exists(self.depths_path)
            and os.path.exists(self.ids_path)
        )

        if self.mode != "test":
            cache_exists = cache_exists and os.path.exists(self.masks_path)

        if load_cached_data and cache_exists:
            print(
                f"Loading cached {self.mode} data from {os.path.dirname(self.images_path)}..."
            )
            self.images = np.load(self.images_path)
            self.depths = np.load(self.depths_path)
            self.ids = np.load(self.ids_path, allow_pickle=True)
            if self.mode != "test":
                self.masks = np.load(self.masks_path)
            else:
                self.masks = None
        else:
            print(f"Processing {self.mode} data from scratch...")
            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError(
                    f"Metadata file not found: {self.metadata_path}"
                )

            df = pd.read_csv(self.metadata_path)

            images_list = []
            masks_list = []
            depths_list = []
            ids_list = []

            for idx, row in df.iterrows():
                # Load Image
                # Metadata paths are relative, e.g., "train/images/xxxx.png"
                img_path = os.path.join(self.root_dir, row["image_path"])
                # Load as grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(f"Image not found: {img_path}")

                images_list.append(img)
                depths_list.append(row["z"])
                ids_list.append(row["id"])

                if self.mode != "test":
                    mask_path = os.path.join(self.root_dir, row["mask_path"])
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise FileNotFoundError(f"Mask not found: {mask_path}")
                    # Binarize mask (0 or 255 -> 0 or 1)
                    mask = (mask > 127).astype(np.uint8)
                    masks_list.append(mask)

            # Convert to numpy arrays
            self.images = np.array(images_list, dtype=np.uint8)
            self.depths = np.array(depths_list, dtype=np.float32)
            self.ids = np.array(ids_list)

            # Save to cache
            np.save(self.images_path, self.images)
            np.save(self.depths_path, self.depths)
            np.save(self.ids_path, self.ids)

            if self.mode != "test":
                self.masks = np.array(masks_list, dtype=np.uint8)
                np.save(self.masks_path, self.masks)
            else:
                self.masks = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Get raw data
        img = self.images[idx]  # Shape: (101, 101), dtype: uint8
        depth = self.depths[idx]  # Scalar float
        img_id = self.ids[idx]

        # 2. Preprocessing
        # Pad image to 128x128 using reflection padding
        img_padded = pad_image_128(img)  # Shape: (128, 128)

        # Normalize Image [0, 1]
        img_normalized = img_padded.astype(np.float32) / 255.0

        # Normalize Depth
        depth_normalized = (depth - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)

        # Create Depth Channel
        # Create a dense channel filled with the normalized depth value
        depth_channel = np.full_like(img_normalized, depth_normalized)

        # Stack to create input tensor
        # Result Shape: (2, 128, 128) -> Channel 0: Image, Channel 1: Depth
        image_tensor = np.stack([img_normalized, depth_channel], axis=0)

        # Handle Mask
        mask_tensor = None
        if self.mode != "test":
            mask = self.masks[idx]  # Shape: (101, 101), dtype: uint8
            mask_padded = pad_image_128(mask)  # Shape: (128, 128)
            # Normalize to 0/1 float
            mask_float = mask_padded.astype(np.float32)
            # Expand dims to (1, 128, 128)
            mask_tensor = mask_float[np.newaxis, :, :]

        # 3. Augmentation (Training Only)
        # Horizontal Flip
        if self.mode == "train" and self.transform:
            if np.random.rand() < 0.5:
                # Flip width (axis 2 for (C, H, W))
                # Use .copy() to avoid negative stride issues with PyTorch
                image_tensor = np.flip(image_tensor, axis=2).copy()
                if mask_tensor is not None:
                    mask_tensor = np.flip(mask_tensor, axis=2).copy()

        # Convert to Torch Tensors
        image_torch = torch.from_numpy(image_tensor).float()

        if self.mode != "test":
            mask_torch = torch.from_numpy(mask_tensor).float()
            return image_torch, mask_torch, img_id
        else:
            return image_torch, img_id


def get_dataloaders(batch_size=32, num_workers=2, load_cached_data=True):
    """
    Factory function to create DataLoaders for train, val, and test sets.

    Args:
        batch_size (int): Batch size for loaders.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Metadata paths
    train_meta = "./metadata/train.csv"
    val_meta = "./metadata/val.csv"
    test_meta = "./metadata/test.csv"

    # Create Datasets
    # Train: Apply augmentation (transform=True)
    train_dataset = SaltDataset(
        train_meta, mode="train", transform=True, load_cached_data=load_cached_data
    )

    # Val: No augmentation
    val_dataset = SaltDataset(
        val_meta, mode="val", transform=False, load_cached_data=load_cached_data
    )

    # Test: No augmentation
    test_dataset = SaltDataset(
        test_meta, mode="test", transform=False, load_cached_data=load_cached_data
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
