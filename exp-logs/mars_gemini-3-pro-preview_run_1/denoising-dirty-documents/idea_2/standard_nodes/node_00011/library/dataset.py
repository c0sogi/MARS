import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class DenoisingDataset(Dataset):
    """
    Dataset class for loading and preprocessing noisy and clean text images.
    Handles RAM caching, random cropping, and geometric augmentations.
    """

    def __init__(self, mode, csv_file, root_dir=Config.INPUT_DIR, debug=Config.DEBUG):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            csv_file (str): Path to the metadata CSV file.
            root_dir (str): Root directory containing the images.
            debug (bool): If True, limits the dataset to a few samples for debugging.
        """
        self.mode = mode
        self.root_dir = root_dir
        self.patch_size = Config.PATCH_SIZE

        # Load metadata
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Metadata file not found: {csv_file}")

        self.df = pd.read_csv(csv_file)

        # Handle Debugging
        if debug:
            self.df = self.df.head(Config.DEBUG_SAMPLES)

        # Preload images into memory (RAM Cache)
        # The dataset is small enough (~100MB raw) that this provides significant speedup
        self.data = []
        for _, row in self.df.iterrows():
            entry = {"id": str(row["id"])}

            # Load Noisy Image
            noisy_path = os.path.join(self.root_dir, row["noisy_image_path"])
            img_n = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)

            if img_n is None:
                continue  # Skip if file is missing/corrupt

            entry["noisy"] = img_n

            # Load Clean Image (only for train/val)
            if self.mode != "test":
                clean_path = os.path.join(self.root_dir, row["clean_image_path"])
                img_c = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)

                if img_c is None:
                    continue

                entry["clean"] = img_c

            self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Get raw images
        img_n = entry["noisy"]

        # Normalize to [0, 1] float32
        img_n = img_n.astype(np.float32) / 255.0

        if self.mode == "test":
            # Test Mode: Return full image and ID
            # Convert to Tensor (C, H, W) -> (1, H, W)
            tensor_n = torch.from_numpy(img_n).unsqueeze(0)
            return tensor_n, entry["id"]

        else:
            # Train/Val Mode: Retrieve target
            img_c = entry["clean"]
            img_c = img_c.astype(np.float32) / 255.0

            if self.mode == "train":
                # --- Patch-Based Training with Augmentation ---

                h, w = img_n.shape

                # 1. Pad if image is smaller than patch size
                pad_h = max(0, self.patch_size - h)
                pad_w = max(0, self.patch_size - w)

                if pad_h > 0 or pad_w > 0:
                    img_n = np.pad(img_n, ((0, pad_h), (0, pad_w)), mode="reflect")
                    img_c = np.pad(img_c, ((0, pad_h), (0, pad_w)), mode="reflect")
                    h, w = img_n.shape  # Update dims

                # 2. Random Crop
                top = np.random.randint(0, h - self.patch_size + 1)
                left = np.random.randint(0, w - self.patch_size + 1)

                img_n = img_n[
                    top : top + self.patch_size, left : left + self.patch_size
                ]
                img_c = img_c[
                    top : top + self.patch_size, left : left + self.patch_size
                ]

                # 3. Geometric Augmentations
                # Random Horizontal Flip
                if np.random.rand() < 0.5:
                    img_n = np.fliplr(img_n)
                    img_c = np.fliplr(img_c)

                # Random Vertical Flip
                if np.random.rand() < 0.5:
                    img_n = np.flipud(img_n)
                    img_c = np.flipud(img_c)

                # Random 90-degree Rotation
                k = np.random.randint(0, 4)
                if k > 0:
                    img_n = np.rot90(img_n, k)
                    img_c = np.rot90(img_c, k)

                # Fix negative strides from numpy flips for PyTorch
                img_n = np.ascontiguousarray(img_n)
                img_c = np.ascontiguousarray(img_c)

            # Convert to Tensor (C, H, W)
            tensor_n = torch.from_numpy(img_n).unsqueeze(0)
            tensor_c = torch.from_numpy(img_c).unsqueeze(0)

            return tensor_n, tensor_c


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE,
    val_batch_size=1,
    test_batch_size=1,
    num_workers=Config.NUM_WORKERS,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        train_batch_size (int): Batch size for training (patches).
        val_batch_size (int): Batch size for validation (full images). Must be 1 usually.
        test_batch_size (int): Batch size for testing (full images). Must be 1 usually.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Initialize Datasets
    train_ds = DenoisingDataset(mode="train", csv_file=Config.TRAIN_CSV)
    val_ds = DenoisingDataset(mode="val", csv_file=Config.VAL_CSV)
    test_ds = DenoisingDataset(mode="test", csv_file=Config.TEST_CSV)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,  # Use all data even if last batch is small
    )

    # Validation and Test loaders use batch_size=1 because full images have variable sizes
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
