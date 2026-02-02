import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomCrop(
                    height=Config.PATCH_SIZE, width=Config.PATCH_SIZE, always_apply=True
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ],
            additional_targets={"target": "image"},
        )
    else:
        # For validation and test, we just convert to tensor.
        # No cropping or geometric augs.
        return A.Compose([ToTensorV2()], additional_targets={"target": "image"})


class TextDenoisingDataset(Dataset):
    """
    Dataset class for Text Denoising.
    Handles loading, caching, normalization, and high-density patch sampling.
    """

    def __init__(
        self, metadata_path, mode="train", transform=None, load_cached_data=True
    ):
        """
        Args:
            metadata_path (str): Path to the CSV metadata file.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            load_cached_data (bool): Whether to use cached .npy files.
        """
        self.metadata_path = metadata_path
        self.mode = mode
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Set sampling density
        if self.mode == "train":
            self.patches_per_epoch = Config.PATCHES_PER_EPOCH
        else:
            self.patches_per_epoch = 1

        # Load Metadata
        self.df = pd.read_csv(self.metadata_path)
        self.ids = self.df["id"].tolist()
        self.feature_paths = self.df["feature_path"].tolist()

        # Label paths exist only for train/val
        if "label_path" in self.df.columns:
            self.label_paths = self.df["label_path"].tolist()
        else:
            self.label_paths = None

        # In-memory storage
        self.data_noisy = []
        self.data_clean = []

        # Initialize Cache Directory
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Load Data
        self._load_and_process_data()

    def _load_and_process_data(self):
        """
        Iterates through metadata, checks cache, loads or processes images,
        and populates self.data_noisy and self.data_clean.
        """
        for idx, img_id in enumerate(self.ids):
            # Define Cache Paths
            noisy_cache_path = os.path.join(Config.CACHE_DIR, f"{img_id}_noisy.npy")
            clean_cache_path = (
                os.path.join(Config.CACHE_DIR, f"{img_id}_clean.npy")
                if self.label_paths
                else None
            )

            # --- Load Noisy Image ---
            noisy_img = self._get_image(
                self.feature_paths[idx], noisy_cache_path, self.load_cached_data
            )
            self.data_noisy.append(noisy_img)

            # --- Load Clean Image (if available) ---
            if self.label_paths:
                clean_img = self._get_image(
                    self.label_paths[idx], clean_cache_path, self.load_cached_data
                )
                self.data_clean.append(clean_img)
            else:
                self.data_clean.append(None)

    def _get_image(self, rel_path, cache_path, load_cached):
        """
        Retrieves an image either from cache or by processing the raw file.

        Args:
            rel_path (str): Relative path to the image in input dir.
            cache_path (str): Path to the cached .npy file.
            load_cached (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: Normalized image array (H, W, 1) float32.
        """
        # 1. Try loading from cache
        if load_cached and os.path.exists(cache_path):
            try:
                img = np.load(cache_path)
                return img
            except Exception:
                pass  # Fallback to processing if load fails

        # 2. Process from scratch
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        # Read as grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {full_path}")

        # Normalize to [0, 1] and convert to float32
        img = img.astype(np.float32) / 255.0

        # Expand dims to (H, W, 1) for consistency with Albumentations/PyTorch
        img = np.expand_dims(img, axis=-1)

        # Save to cache
        np.save(cache_path, img)

        return img

    def __len__(self):
        """
        Returns the effective length of the dataset.
        For training, this is num_images * patches_per_epoch.
        """
        return len(self.ids) * self.patches_per_epoch

    def __getitem__(self, idx):
        """
        Retrieves an item.

        Args:
            idx (int): Index.

        Returns:
            tuple:
                - Train: (noisy_patch, clean_patch)
                - Val: (noisy_img, clean_img, img_id)
                - Test: (noisy_img, img_id)
        """
        # Map linear index to image index
        img_idx = idx // self.patches_per_epoch

        img_id = self.ids[img_idx]
        noisy = self.data_noisy[img_idx]
        clean = self.data_clean[img_idx]

        # Apply Transforms
        if self.transform:
            if self.mode == "train" or self.mode == "val":
                # Paired transformation
                augmented = self.transform(image=noisy, target=clean)
                noisy_out = augmented["image"]
                clean_out = augmented["target"]
            else:
                # Test mode (no target)
                augmented = self.transform(image=noisy)
                noisy_out = augmented["image"]
                clean_out = None
        else:
            # Fallback to simple tensor conversion if no transform provided
            noisy_out = torch.from_numpy(noisy.transpose(2, 0, 1))
            clean_out = (
                torch.from_numpy(clean.transpose(2, 0, 1))
                if clean is not None
                else None
            )

        # Return based on mode
        if self.mode == "train":
            return noisy_out, clean_out
        elif self.mode == "val":
            return noisy_out, clean_out, img_id
        else:
            return noisy_out, img_id
