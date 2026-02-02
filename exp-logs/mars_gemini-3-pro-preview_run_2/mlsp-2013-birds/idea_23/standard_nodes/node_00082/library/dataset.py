import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG


def get_transforms(mode="train"):
    """
    Returns the Albumentations composition for the given mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                # Resize is handled during caching/loading to ensure consistent dimensions for batching
                # But we include it here just in case raw images are passed or for safety
                A.Resize(height=CFG.img_height, width=CFG.img_width),
                # Standard Augmentations
                A.RandomBrightnessContrast(p=0.5),
                # SpecAugment simulation using CoarseDropout
                # Masking out rectangular regions in time/frequency
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(CFG.img_height * 0.1),
                    max_width=int(CFG.img_width * 0.1),
                    min_holes=2,
                    fill_value=0,
                    p=0.5,
                ),
                # Normalize to ImageNet stats
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # Convert to Tensor
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / Inference
        return A.Compose(
            [
                A.Resize(height=CFG.img_height, width=CFG.img_width),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def load_and_cache_images(df, cache_name, load_cached_data=True):
    """
    Loads images from disk, resizes them, and caches them as a numpy array.

    Args:
        df (pd.DataFrame): DataFrame containing file paths.
        cache_name (str): Unique identifier for the cache file (e.g., 'train_images').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Array of shape (N, H, W, 3) containing loaded images.
    """
    cache_path = os.path.join(CFG.output_dir, f"{cache_name}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # print(f"Loading cached images from {cache_path}...")
            images = np.load(cache_path)
            return images
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    # print(f"Processing images for {cache_name}...")
    images = []

    for _, row in df.iterrows():
        # Construct path to filtered spectrogram
        # Metadata points to standard spectrograms, we need to redirect to filtered ones
        original_spec_path = row["file_path_spec"]
        filename = os.path.basename(original_spec_path)
        full_path = os.path.join(CFG.filtered_spectrogram_dir, filename)

        # Load image
        if not os.path.exists(full_path):
            # Fallback to input dir if relative path logic fails or file missing
            # Try constructing full path relative to ./input if provided path is relative
            full_path = os.path.join(
                "./input", "supplemental_data", "filtered_spectrograms", filename
            )

        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Create a black image if file is missing (should not happen based on EDA)
            img = np.zeros((CFG.img_height, CFG.img_width), dtype=np.uint8)
        else:
            # Resize to target dimensions
            img = cv2.resize(img, (CFG.img_width, CFG.img_height))

        # Convert to Pseudo-RGB (H, W) -> (H, W, 3)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        images.append(img)

    images = np.array(images, dtype=np.uint8)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, images)
    # print(f"Cached images to {cache_path}")

    return images


class BirdDataset(Dataset):
    """
    Dataset class for Bird Species Classification.
    Handles loading, caching, time-rolling, and augmentations.
    """

    def __init__(self, df, mode="train", tta_shift=0.0, load_cached_data=True):
        """
        Args:
            df (pd.DataFrame): DataFrame with metadata.
            mode (str): 'train', 'val', or 'test'.
            tta_shift (float): Fraction of width to roll the image along time axis (0.0 to 1.0).
            load_cached_data (bool): Whether to use cached image arrays.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.tta_shift = tta_shift

        # Identify label columns
        self.label_cols = [c for c in df.columns if c.startswith("species_")]
        self.labels = self.df[self.label_cols].values.astype(np.float32)

        # Load images (cached)
        # We create a unique cache name based on the dataframe content hash or length+mode
        # For simplicity in this contest environment, we use mode + length
        cache_name = f"images_{mode}_{len(df)}"
        self.images = load_and_cache_images(df, cache_name, load_cached_data)

        self.transforms = get_transforms(mode)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve image from memory
        img = self.images[idx].copy()  # (H, W, 3)

        # Apply Cyclic Time-Rolling
        # 1. Deterministic TTA shift (if provided)
        if self.tta_shift > 0:
            shift_pixels = int(CFG.img_width * self.tta_shift)
            img = np.roll(img, shift_pixels, axis=1)  # Axis 1 is Width (Time)

        # 2. Random Time-Rolling during training
        elif self.mode == "train":
            # Random shift
            shift_factor = np.random.rand()
            shift_pixels = int(CFG.img_width * shift_factor)
            img = np.roll(img, shift_pixels, axis=1)

        # Apply Albumentations
        # Albumentations expects RGB images
        augmented = self.transforms(image=img)
        image_tensor = augmented["image"]

        # Get labels
        label = torch.tensor(self.labels[idx])

        return image_tensor, label
