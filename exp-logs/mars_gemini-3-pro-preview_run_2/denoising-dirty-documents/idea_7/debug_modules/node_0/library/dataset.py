import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class DenoisingDataset(Dataset):
    """
    Dataset class for the Denoising Task.
    Handles loading of paired noisy/clean images, caching, and high-density patch sampling.
    """

    def __init__(self, metadata_path, mode="train", load_cached_data=True):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load processed data from cache.
        """
        self.mode = mode
        self.metadata_path = metadata_path
        # High-Density Sampling: Expand dataset length for training
        self.patches_per_image = Config.PATCHES_PER_IMAGE if mode == "train" else 1
        self.patch_size = Config.PATCH_SIZE

        # Create cache directory structure
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache")
        if self.mode == "train":
            self.cache_subdir = os.path.join(self.cache_dir, "train")
        elif self.mode == "val":
            self.cache_subdir = os.path.join(self.cache_dir, "val")
        else:
            self.cache_subdir = os.path.join(self.cache_dir, "test")

        os.makedirs(self.cache_subdir, exist_ok=True)

        # Load Metadata
        self.df = pd.read_csv(metadata_path)

        # Debugging Subset
        if Config.DEBUG_SUBSET_SIZE is not None:
            self.df = self.df.iloc[: Config.DEBUG_SUBSET_SIZE]

        # Load Data (with caching logic)
        self.data = self._load_and_cache_data(load_cached_data)

        # Define Augmentations
        if self.mode == "train":
            # High-Density Sampling & Augmentation
            # Random Crop + Geometric Transformations (Flips/Rotations)
            self.transform = A.Compose(
                [
                    A.RandomCrop(height=self.patch_size, width=self.patch_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    ToTensorV2(),
                ],
                additional_targets={"clean": "image"},
            )
        else:
            # Full image for validation/test (no cropping, just tensor conversion)
            self.transform = A.Compose(
                [ToTensorV2()], additional_targets={"clean": "image"}
            )

    def _load_and_cache_data(self, load_cached_data):
        """
        Loads images, normalizes them, and caches them as .npy files.
        Strictly follows the caching logic:
        1. IF load_cached_data is True: Try to load.
        2. IF fail OR False: Compute, Save, Return.
        """
        loaded_data = []

        for _, row in self.df.iterrows():
            img_id = str(row["id"])

            # Define cache paths
            noisy_cache_path = os.path.join(self.cache_subdir, f"{img_id}_noisy.npy")
            clean_cache_path = os.path.join(self.cache_subdir, f"{img_id}_clean.npy")

            # Determine if we have ground truth (train/val) or not (test)
            has_label = "label_path" in row and pd.notna(row["label_path"])

            # --- Attempt to load from cache ---
            noisy_img = None
            clean_img = None

            cache_hit = False
            if load_cached_data:
                # Check if files exist
                if os.path.exists(noisy_cache_path):
                    if has_label:
                        if os.path.exists(clean_cache_path):
                            try:
                                noisy_img = np.load(noisy_cache_path)
                                clean_img = np.load(clean_cache_path)
                                cache_hit = True
                            except Exception:
                                cache_hit = False
                    else:
                        try:
                            noisy_img = np.load(noisy_cache_path)
                            cache_hit = True
                        except Exception:
                            cache_hit = False

            # --- Compute if cache miss ---
            if not cache_hit:
                # Load Noisy Image
                noisy_path = os.path.join(Config.INPUT_DIR, row["feature_path"])
                noisy_raw = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
                if noisy_raw is None:
                    continue  # Skip missing files

                # Normalize to 0-1 float32
                noisy_img = noisy_raw.astype(np.float32) / 255.0

                # Save Noisy to Cache
                np.save(noisy_cache_path, noisy_img)

                # Load Clean Image if applicable
                if has_label:
                    clean_path = os.path.join(Config.INPUT_DIR, row["label_path"])
                    clean_raw = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
                    if clean_raw is None:
                        continue

                    clean_img = clean_raw.astype(np.float32) / 255.0
                    np.save(clean_cache_path, clean_img)

            # Store in memory
            # Add channel dimension for Albumentations (H, W) -> (H, W, 1)
            # This ensures ToTensorV2 creates (1, H, W)
            if noisy_img is not None:
                noisy_img = np.expand_dims(noisy_img, axis=2)

            if clean_img is not None:
                clean_img = np.expand_dims(clean_img, axis=2)

            item = {"id": img_id, "noisy": noisy_img}
            if clean_img is not None:
                item["clean"] = clean_img

            loaded_data.append(item)

        return loaded_data

    def __len__(self):
        """
        Returns the length of the dataset.
        For training, length is num_images * patches_per_image (High-Density Sampling).
        For val/test, length is num_images.
        """
        return len(self.data) * self.patches_per_image

    def __getitem__(self, idx):
        """
        Retrieves a sample.
        Args:
            idx (int): Index.
        Returns:
            tuple: (noisy_tensor, clean_tensor, id) for train/val.
                   (noisy_tensor, id) for test.
        """
        # Map linear index to image index
        image_idx = idx // self.patches_per_image
        sample = self.data[image_idx]

        noisy = sample["noisy"]
        img_id = sample["id"]

        if "clean" in sample:
            clean = sample["clean"]

            # Apply transforms (Augmentation / ToTensor)
            # Pass clean image as 'clean' target to ensure identical geometric transforms
            augmented = self.transform(image=noisy, clean=clean)

            noisy_tensor = augmented["image"]
            clean_tensor = augmented["clean"]

            return noisy_tensor, clean_tensor, img_id
        else:
            # Test mode (no clean image)
            augmented = self.transform(image=noisy)
            noisy_tensor = augmented["image"]

            return noisy_tensor, img_id
