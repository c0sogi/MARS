import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class DenoisingDataset(Dataset):
    """
    Dataset class for loading and preprocessing noisy and clean text images.
    """

    def __init__(self, metadata_path, mode="train", limit=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'. Controls preprocessing and return values.
            limit (int, optional): Limit the dataset size for debugging.
        """
        self.mode = mode
        self.metadata_path = metadata_path
        self.input_dir = Config.INPUT_DIR
        self.patch_size = Config.PATCH_SIZE

        # Load metadata
        self.df = pd.read_csv(metadata_path)

        # Optional: Limit dataset size for debugging
        if limit is not None:
            self.df = self.df.iloc[:limit]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row["id"])

        # --- Load Noisy Image (Input) ---
        noisy_path = os.path.join(self.input_dir, row["noisy_image_path"])
        # Load in grayscale
        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        if noisy_img is None:
            raise FileNotFoundError(f"Image not found at {noisy_path}")

        # Normalize to [0, 1]
        noisy_img = noisy_img.astype(np.float32) / 255.0

        # --- Test Mode ---
        if self.mode == "test":
            # Return full image and ID for submission
            # Add channel dimension: (H, W) -> (1, H, W)
            noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)
            return noisy_tensor, img_id

        # --- Load Clean Image (Target) for Train/Val ---
        clean_path = os.path.join(self.input_dir, row["clean_image_path"])
        clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
        if clean_img is None:
            raise FileNotFoundError(f"Image not found at {clean_path}")

        # Normalize to [0, 1]
        clean_img = clean_img.astype(np.float32) / 255.0

        # --- Training Mode: Random Crop ---
        if self.mode == "train":
            h, w = noisy_img.shape

            # Pad if image is smaller than patch size
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)

            if pad_h > 0 or pad_w > 0:
                # Pad right and bottom with reflection
                noisy_img = np.pad(noisy_img, ((0, pad_h), (0, pad_w)), mode="reflect")
                clean_img = np.pad(clean_img, ((0, pad_h), (0, pad_w)), mode="reflect")
                h, w = noisy_img.shape

            # Random coordinates for crop
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)

            # Extract patches
            noisy_patch = noisy_img[
                top : top + self.patch_size, left : left + self.patch_size
            ]
            clean_patch = clean_img[
                top : top + self.patch_size, left : left + self.patch_size
            ]

            # --- Augmentation ---
            # Random horizontal flip
            if np.random.rand() < 0.5:
                noisy_patch = np.fliplr(noisy_patch)
                clean_patch = np.fliplr(clean_patch)

            # Random vertical flip
            if np.random.rand() < 0.5:
                noisy_patch = np.flipud(noisy_patch)
                clean_patch = np.flipud(clean_patch)

            # Random rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                noisy_patch = np.rot90(noisy_patch, k)
                clean_patch = np.rot90(clean_patch, k)

            # Ensure contiguous arrays for torch
            noisy_patch = np.ascontiguousarray(noisy_patch)
            clean_patch = np.ascontiguousarray(clean_patch)

            # Convert to tensor (C, H, W)
            noisy_tensor = torch.from_numpy(noisy_patch).unsqueeze(0)
            clean_tensor = torch.from_numpy(clean_patch).unsqueeze(0)

            return noisy_tensor, clean_tensor

        # --- Validation Mode: Full Image ---
        else:
            # Return full images
            noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)
            clean_tensor = torch.from_numpy(clean_img).unsqueeze(0)
            return noisy_tensor, clean_tensor


def get_dataloaders(dataset_limit=None):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        dataset_limit (int, optional): Limit the number of samples for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Train Dataset & Loader
    # Uses random cropping and batching
    train_ds = DenoisingDataset(
        Config.TRAIN_METADATA_PATH, mode="train", limit=dataset_limit
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Validation Dataset & Loader
    # Uses batch_size=1 because full images have variable dimensions
    val_ds = DenoisingDataset(Config.VAL_METADATA_PATH, mode="val", limit=dataset_limit)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Test Dataset & Loader
    # Uses batch_size=1 for variable dimensions
    test_ds = DenoisingDataset(
        Config.TEST_METADATA_PATH, mode="test", limit=dataset_limit
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
