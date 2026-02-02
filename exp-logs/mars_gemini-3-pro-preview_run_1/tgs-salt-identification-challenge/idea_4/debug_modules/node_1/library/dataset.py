import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2

# Import from provided library files
from library.config import Config
from library.utils import load_data_and_cache, pad_image


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles on-the-fly preprocessing:
    - Normalization (Image and Depth)
    - Depth fusion (Concatenating depth as a channel)
    - Reflection Padding (101 -> 128)
    - Augmentation (Random Horizontal Flip for training)
    """

    def __init__(self, images, depths, masks=None, mode="train"):
        """
        Args:
            images (np.ndarray): Array of images (N, 101, 101, 1).
            depths (np.ndarray): Array of depths (N,).
            masks (np.ndarray, optional): Array of masks (N, 101, 101, 1).
            mode (str): 'train', 'val', or 'test'. Controls augmentation.
        """
        self.images = images
        self.depths = depths
        self.masks = masks
        self.mode = mode

        # Target size for padding (from Config)
        self.target_size = Config.IMG_SIZE  # 128

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load data
        # Image is (101, 101, 1), uint8
        img = self.images[idx]
        depth = self.depths[idx]

        # Initialize mask
        if self.masks is not None:
            mask = self.masks[idx]
        else:
            mask = None

        # 2. Augmentation (Random Horizontal Flip)
        # Apply to original size before padding/fusion to be safe
        if self.mode == "train":
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1)  # Flip width dimension

                if mask is not None:
                    mask = np.flip(mask, axis=1)

        # 3. Preprocessing

        # Normalize Image: (0-255) -> (0-1)
        img = img.astype(np.float32) / 255.0

        # Normalize Depth: Simple scaling based on domain knowledge (max ~1000)
        # This keeps depth values roughly in the same 0-1 range as pixels
        depth_norm = float(depth) / 1000.0

        # Pad Image: (101, 101, 1) -> (128, 128, 1) using Reflection Padding
        img_padded = pad_image(img, target_size=self.target_size)

        # Create Depth Channel: (128, 128, 1)
        # We create a dense channel where every pixel has the normalized depth value
        depth_channel = np.full(
            (self.target_size, self.target_size, 1), depth_norm, dtype=np.float32
        )

        # Fuse Channels: (128, 128, 1) + (128, 128, 1) -> (128, 128, 2)
        img_fused = np.concatenate([img_padded, depth_channel], axis=-1)

        # Transpose to Channel-First: (H, W, C) -> (C, H, W) => (2, 128, 128)
        img_tensor = torch.from_numpy(img_fused.transpose(2, 0, 1)).float()

        # Handle Mask
        if mask is not None:
            # Pad Mask: (101, 101, 1) -> (128, 128, 1)
            mask_padded = pad_image(mask, target_size=self.target_size)

            # Ensure binary float (0.0 or 1.0)
            mask_padded = (mask_padded > 0).astype(np.float32)

            # Transpose: (1, 128, 128)
            mask_tensor = torch.from_numpy(mask_padded.transpose(2, 0, 1)).float()
            return img_tensor, mask_tensor

        return img_tensor


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    load_cached_data=True,
):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    Implements the 90/10 stratified split strategy on the combined training data.

    Args:
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of subprocesses for data loading.
        debug (bool): If True, subsamples data for rapid debugging.
        load_cached_data (bool): If True, attempts to load processed arrays from disk.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # -------------------------------------------------------------------------
    # 1. Load Metadata
    # -------------------------------------------------------------------------
    # We load both train and val metadata files provided by the system
    df_train_meta = pd.read_csv(Config.TRAIN_CSV)
    df_val_meta = pd.read_csv(Config.VAL_CSV)
    df_test_meta = pd.read_csv(Config.TEST_CSV)

    # Combine original train/val to perform a new, single 90/10 split
    # as per the "Idea 4" strategy (abandoning 5-fold CV).
    df_full_train = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # -------------------------------------------------------------------------
    # 2. Debugging Subsampling
    # -------------------------------------------------------------------------
    if debug:
        print(f"DEBUG MODE: Subsampling {Config.DEBUG_SIZE} samples.")
        df_full_train = df_full_train.sample(
            n=min(len(df_full_train), Config.DEBUG_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

        df_test_meta = df_test_meta.sample(
            n=min(len(df_test_meta), Config.DEBUG_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 3. Load Data (Caching)
    # -------------------------------------------------------------------------
    # Load all training data into numpy arrays
    # This function handles reading images/masks from disk and caching them as .npy
    train_data = load_data_and_cache(
        df_full_train, dataset_type="combined_train", load_cached_data=load_cached_data
    )

    # Load test data
    test_data = load_data_and_cache(
        df_test_meta, dataset_type="test", load_cached_data=load_cached_data
    )

    # -------------------------------------------------------------------------
    # 4. Stratified Split (90/10)
    # -------------------------------------------------------------------------
    # We use the 'coverage_class' column for stratification to ensure salt distribution is balanced
    X_indices = np.arange(len(df_full_train))
    y_strat = df_full_train["coverage_class"].values

    # If debug mode makes classes too small for stratification, fall back to random
    try:
        train_idx, val_idx = train_test_split(
            X_indices, test_size=0.1, random_state=Config.SEED, stratify=y_strat
        )
    except ValueError:
        # Fallback for debug mode if classes are missing or too small
        train_idx, val_idx = train_test_split(
            X_indices, test_size=0.1, random_state=Config.SEED
        )

    # -------------------------------------------------------------------------
    # 5. Create Datasets
    # -------------------------------------------------------------------------

    # Training Dataset (90%)
    train_dataset = SaltDataset(
        images=train_data["images"][train_idx],
        depths=train_data["depths"][train_idx],
        masks=train_data["masks"][train_idx],
        mode="train",
    )

    # Validation Dataset (10%)
    val_dataset = SaltDataset(
        images=train_data["images"][val_idx],
        depths=train_data["depths"][val_idx],
        masks=train_data["masks"][val_idx],
        mode="val",
    )

    # Test Dataset
    test_dataset = SaltDataset(
        images=test_data["images"], depths=test_data["depths"], masks=None, mode="test"
    )

    # -------------------------------------------------------------------------
    # 6. Create DataLoaders
    # -------------------------------------------------------------------------
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
