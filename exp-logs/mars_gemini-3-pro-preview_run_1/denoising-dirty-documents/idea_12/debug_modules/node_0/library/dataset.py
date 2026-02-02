import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_or_process_cache


def _load_images_from_csv(csv_path, input_dir):
    """
    Reads images defined in the CSV and returns a dictionary of numpy arrays.
    Used as the processing function for the caching mechanism.

    Args:
        csv_path (str): Path to the metadata CSV.
        input_dir (str): Root directory for images.

    Returns:
        dict: Dictionary mapping keys (e.g., 'noisy_101') to image arrays.
    """
    df = pd.read_csv(csv_path)
    data_dict = {}

    # Iterate through metadata to load images
    for _, row in df.iterrows():
        img_id = str(row["id"])

        # Load Noisy Image
        noisy_path = os.path.join(input_dir, row["noisy_image_path"])
        if not os.path.exists(noisy_path):
            raise FileNotFoundError(f"Noisy image not found: {noisy_path}")

        # Load as grayscale (H, W)
        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        if noisy_img is None:
            raise ValueError(f"Failed to load image: {noisy_path}")

        data_dict[f"noisy_{img_id}"] = noisy_img

        # Load Clean Image (if available)
        if "clean_image_path" in row and pd.notna(row["clean_image_path"]):
            clean_path = os.path.join(input_dir, row["clean_image_path"])
            if os.path.exists(clean_path):
                clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
                if clean_img is None:
                    raise ValueError(f"Failed to load image: {clean_path}")
                data_dict[f"clean_{img_id}"] = clean_img
            else:
                raise FileNotFoundError(f"Clean image not found: {clean_path}")

    return data_dict


class DenoisingDataset(Dataset):
    """
    Dataset class for the Denoising Task.
    Handles Signal Inversion, Reflection Padding, and Augmentations.
    """

    def __init__(self, data_dict, csv_path, mode="train"):
        """
        Args:
            data_dict (dict): Dictionary containing loaded image arrays.
            csv_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
        """
        self.data_dict = data_dict
        self.mode = mode
        self.df = pd.read_csv(csv_path)
        # Ensure IDs are strings for consistent lookup
        self.ids = self.df["id"].astype(str).tolist()

        # --- Augmentation Pipeline ---
        if self.mode == "train":
            self.transform = A.Compose(
                [
                    # Ensure image is at least PATCH_SIZE x PATCH_SIZE using Reflection Padding
                    A.PadIfNeeded(
                        min_height=Config.PATCH_SIZE,
                        min_width=Config.PATCH_SIZE,
                        border_mode=cv2.BORDER_REFLECT,
                    ),
                    # Random Crop to fixed size for batching
                    A.RandomCrop(height=Config.PATCH_SIZE, width=Config.PATCH_SIZE),
                    # Geometric Augmentations
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    ToTensorV2(),
                ]
            )
        else:
            # Validation/Test: No cropping, but pad to multiples of 16 for U-Net architecture
            self.transform = A.Compose(
                [
                    A.PadIfNeeded(
                        pad_height_divisor=16,
                        pad_width_divisor=16,
                        border_mode=cv2.BORDER_REFLECT,
                    ),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # --- 1. Retrieve Raw Data ---
        noisy_key = f"noisy_{img_id}"
        noisy_img = self.data_dict[noisy_key]

        # --- 2. Preprocessing (Normalize & Invert) ---
        # Normalize to [0, 1] float
        noisy_img = noisy_img.astype(np.float32) / 255.0

        # Signal Inversion: Map background (white) to 0, text (black) to 1
        if Config.INVERT_SIGNAL:
            noisy_img = 1.0 - noisy_img

        # Process Target (Clean Image) if available
        clean_img = None
        clean_key = f"clean_{img_id}"
        if clean_key in self.data_dict:
            clean_img = self.data_dict[clean_key]
            clean_img = clean_img.astype(np.float32) / 255.0
            if Config.INVERT_SIGNAL:
                clean_img = 1.0 - clean_img

        # --- 3. Augmentation ---
        # Albumentations expects 'image' and optional 'mask'
        if clean_img is not None:
            augmented = self.transform(image=noisy_img, mask=clean_img)
            noisy_tensor = augmented["image"]
            clean_tensor = augmented["mask"]
        else:
            augmented = self.transform(image=noisy_img)
            noisy_tensor = augmented["image"]
            # Create dummy target for test set
            clean_tensor = torch.zeros_like(noisy_tensor)

        # --- 4. Channel Dimension Adjustment ---
        # ToTensorV2 converts (H, W) -> (H, W) tensor if input is 2D.
        # We need (1, H, W) for PyTorch Conv2d.
        if noisy_tensor.ndim == 2:
            noisy_tensor = noisy_tensor.unsqueeze(0)

        if clean_tensor.ndim == 2:
            clean_tensor = clean_tensor.unsqueeze(0)

        return noisy_tensor, clean_tensor, img_id


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    Handles caching of raw image data to speed up initialization.

    Args:
        load_cached_data (bool): Whether to attempt loading from existing cache files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # --- 1. Load Data into Memory (with Caching) ---
    print("Initializing DataLoaders...")

    train_cache = load_or_process_cache(
        "train_cache.npz",
        _load_images_from_csv,
        load_cached_data=load_cached_data,
        csv_path=Config.TRAIN_CSV,
        input_dir=Config.INPUT_DIR,
    )

    val_cache = load_or_process_cache(
        "val_cache.npz",
        _load_images_from_csv,
        load_cached_data=load_cached_data,
        csv_path=Config.VAL_CSV,
        input_dir=Config.INPUT_DIR,
    )

    test_cache = load_or_process_cache(
        "test_cache.npz",
        _load_images_from_csv,
        load_cached_data=load_cached_data,
        csv_path=Config.TEST_CSV,
        input_dir=Config.INPUT_DIR,
    )

    # --- 2. Create Datasets ---
    train_ds = DenoisingDataset(train_cache, Config.TRAIN_CSV, mode="train")
    val_ds = DenoisingDataset(val_cache, Config.VAL_CSV, mode="val")
    test_ds = DenoisingDataset(test_cache, Config.TEST_CSV, mode="test")

    # --- 3. Create DataLoaders ---
    # Deterministic worker initialization
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    # Validation and Test use batch_size=1 to handle variable full-image sizes
    # without needing to crop or resize, ensuring accurate metric calculation.
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
