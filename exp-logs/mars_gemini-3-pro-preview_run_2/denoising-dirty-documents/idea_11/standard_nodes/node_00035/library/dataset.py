import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset

from library.config import Config
from library.utils import normalize_image


class DenoisingDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        root_dir,
        mode="train",
        load_cached_data=True,
        limit_size=None,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            root_dir (str): Root directory containing the images (usually ./input).
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to try loading from cache.
            limit_size (int, optional): Limit dataset size for debugging.
        """
        self.metadata_path = metadata_path
        self.root_dir = root_dir
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.cache_dir = Config.CACHE_DIR
        self.patch_size = Config.PATCH_SIZE
        self.patches_per_image = Config.PATCHES_PER_IMAGE

        # Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        if limit_size is not None:
            self.df = self.df.iloc[:limit_size]

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Pre-load data into memory
        self.data = []
        self._load_and_process_data()

        # Define Augmentations for Training
        # We use additional_targets to ensure the same augmentation is applied to both noisy and clean patches
        if self.mode == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ],
                additional_targets={"image1": "image"},
            )

    def _load_and_process_data(self):
        """
        Iterates through metadata, loads images (from cache or source),
        normalizes them, and stores them in memory.
        """
        for _, row in self.df.iterrows():
            img_id = str(row["id"])
            sample = {"id": img_id}

            # 1. Load Noisy Image (Feature)
            noisy_rel_path = row["feature_path"]
            noisy_cache_name = f"{img_id}_noisy.npy"
            sample["noisy"] = self._get_image_array(noisy_rel_path, noisy_cache_name)

            # 2. Load Clean Image (Label) - Only for train/val
            if self.mode != "test":
                clean_rel_path = row["label_path"]
                clean_cache_name = f"{img_id}_clean.npy"
                sample["clean"] = self._get_image_array(
                    clean_rel_path, clean_cache_name
                )

            self.data.append(sample)

    def _get_image_array(self, rel_path, cache_name):
        """
        Retrieves image array from cache or processes from source.
        """
        cache_path = os.path.join(self.cache_dir, cache_name)

        # Logic:
        # 1. IF load_cached_data is True: Try to load the file.
        # 2. IF loading fails OR load_cached_data is False: Compute and save.

        if self.load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception as e:
                # If load fails, we proceed to reload from source
                pass

        # Load from source
        full_path = os.path.join(self.root_dir, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")

        # Read as Grayscale
        # IMREAD_GRAYSCALE ensures we get a 2D array (H, W)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {full_path}")

        # Normalize [0, 255] -> [0, 1]
        img = normalize_image(img)

        # Save to cache
        np.save(cache_path, img)

        return img

    def __len__(self):
        """
        Returns the length of the dataset.
        For training, length is artificially expanded by PATCHES_PER_IMAGE
        to implement High-Density Sampling.
        """
        if self.mode == "train":
            return len(self.data) * self.patches_per_image
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a sample.
        """
        if self.mode == "train":
            # Map expanded index to actual image index
            img_idx = idx // self.patches_per_image
            sample = self.data[img_idx]

            noisy = sample["noisy"]
            clean = sample["clean"]
            img_id = sample["id"]

            # Random Crop Logic
            h, w = noisy.shape

            # Handle edge case where image is smaller than patch size
            if h < self.patch_size or w < self.patch_size:
                pad_h = max(0, self.patch_size - h)
                pad_w = max(0, self.patch_size - w)
                noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode="reflect")
                clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
                h, w = noisy.shape

            # Generate random coordinates
            y = np.random.randint(0, h - self.patch_size + 1)
            x = np.random.randint(0, w - self.patch_size + 1)

            # Extract patches
            patch_noisy = noisy[y : y + self.patch_size, x : x + self.patch_size]
            patch_clean = clean[y : y + self.patch_size, x : x + self.patch_size]

            # Apply Augmentations
            # Pass clean image as 'image1' so identical transforms are applied to both
            augmented = self.transform(image=patch_noisy, image1=patch_clean)
            patch_noisy = augmented["image"]
            patch_clean = augmented["image1"]

            # Convert to Tensor (C, H, W)
            # Albumentations returns numpy arrays here since we didn't use ToTensorV2 in Compose
            patch_noisy = torch.from_numpy(patch_noisy).unsqueeze(0).float()
            patch_clean = torch.from_numpy(patch_clean).unsqueeze(0).float()

            return patch_noisy, patch_clean, img_id

        elif self.mode == "val":
            sample = self.data[idx]
            noisy = sample["noisy"]
            clean = sample["clean"]
            img_id = sample["id"]

            # Return full images for validation (batch_size should be 1)
            noisy = torch.from_numpy(noisy).unsqueeze(0).float()
            clean = torch.from_numpy(clean).unsqueeze(0).float()

            return noisy, clean, img_id

        elif self.mode == "test":
            sample = self.data[idx]
            noisy = sample["noisy"]
            img_id = sample["id"]

            # Return full image for testing
            noisy = torch.from_numpy(noisy).unsqueeze(0).float()

            return noisy, img_id
