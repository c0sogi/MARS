import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(phase: str):
    """
    Constructs the Albumentations transform pipeline based on the phase and Config.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    transforms = []

    # Note: Primary resizing is handled in the caching step to 'squash' images
    # deterministically. We add a safety resize here to ensure tensor shape consistency.
    transforms.append(A.Resize(height=Config.image_size, width=Config.image_size))

    if phase == "train":
        # Geometric Augmentations (Strict Invariance)
        if Config.aug_hflip:
            transforms.append(A.HorizontalFlip(p=0.5))
        if Config.aug_vflip:
            transforms.append(A.VerticalFlip(p=0.5))
        if Config.aug_rotate90:
            transforms.append(A.RandomRotate90(p=0.5))

        # Color Jitter is disabled in Config to preserve Graham's Norm features
        if Config.aug_color_jitter:
            transforms.append(A.ColorJitter(p=0.5))

    # Standard Normalization (ImageNet stats)
    transforms.append(
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    )
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def process_and_cache_data(df: pd.DataFrame, phase: str, load_cached_data: bool = True):
    """
    Processes images applying Graham's Normalization and Resizing, then caches
    the result to disk/RAM. Strictly follows the required caching logic.

    Args:
        df (pd.DataFrame): The metadata dataframe.
        phase (str): The dataset phase (train/val/test).
        load_cached_data (bool): Flag to enable/disable loading from cache.

    Returns:
        np.ndarray: Array of processed images (N, H, W, C).
    """
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Append debug suffix to avoid cache collisions during development
    suffix = "_debug" if Config.debug else ""
    cache_filename = f"cached_images_{phase}{suffix}.npy"
    cache_path = os.path.join(Config.working_dir, cache_filename)

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"[{phase}] Loading cached data from {cache_path}...")
            data = np.load(cache_path)

            # Verify consistency
            if len(data) == len(df):
                return data
            else:
                print(
                    f"[{phase}] Cache size mismatch (Expected {len(df)}, got {len(data)}). Recomputing..."
                )
        except Exception as e:
            print(f"[{phase}] Failed to load cache: {e}. Recomputing...")

    # 2. IF loading fails OR load_cached_data is False: Compute and save.
    print(f"[{phase}] Processing and caching {len(df)} images...")

    images = []
    size = Config.image_size

    for _, row in df.iterrows():
        # Construct full path
        full_path = os.path.join(Config.input_root, row["file_path"])

        # Read Image
        img = cv2.imread(full_path)

        if img is None:
            # Fallback for missing/corrupt images (maintaining alignment)
            img = np.zeros((size, size, 3), dtype=np.uint8)
        else:
            # Apply Graham's Color Normalization
            # Formula: 4 * Image - 4 * GaussianBlur(Image) + 128
            if Config.use_graham_norm:
                # cv2.addWeighted handles saturation/clipping to [0, 255] automatically
                img = cv2.addWeighted(
                    img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128
                )

            # Squashing Resize (Simple resize ignoring aspect ratio)
            img = cv2.resize(img, (size, size))

            # Convert BGR (OpenCV) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img)

    # Stack into numpy array
    images_array = np.array(images, dtype=np.uint8)

    # Save to cache
    np.save(cache_path, images_array)
    print(f"[{phase}] Saved processed data to {cache_path}")

    return images_array


class RetinopathyDataset(Dataset):
    """
    Dataset class for Diabetic Retinopathy Detection.
    Handles loading, caching, Graham's preprocessing, and Ordinal Target generation.
    """

    def __init__(
        self, df: pd.DataFrame, phase: str = "train", load_cached_data: bool = True
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            phase (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached preprocessed images.
        """
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.transform = get_transforms(phase)

        # Load preprocessed images into RAM
        self.images = process_and_cache_data(self.df, phase, load_cached_data)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve image from RAM
        image = self.images[idx]

        # Apply Augmentations/Normalization
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Handle Targets
        if "diagnosis" in self.df.columns:
            label = self.df.iloc[idx]["diagnosis"]

            # Ordinal Encoding for Rank-Consistent Regression
            # For K classes, we use K-1 binary heads.
            # Head k predicts Prob(y > k)
            # Example (K=5, Heads=4):
            # Label 0: [0, 0, 0, 0]
            # Label 1: [1, 0, 0, 0]
            # Label 2: [1, 1, 0, 0]
            # Label 4: [1, 1, 1, 1]

            ordinal_target = np.zeros(Config.num_ordinal_heads, dtype=np.float32)
            if label > 0:
                # Set the first 'label' heads to 1
                ordinal_target[:label] = 1.0

            return image, ordinal_target
        else:
            # Inference mode (Test set)
            # Return dummy target
            dummy_target = torch.zeros(Config.num_ordinal_heads, dtype=np.float32)
            return image, dummy_target
