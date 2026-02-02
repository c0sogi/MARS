import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
from library.config import Config


class DenoisingDataset(Dataset):
    """
    Dataset for the Denoising Task.
    Implements High-Density Patching, Caching, and Geometric Augmentations.
    """

    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.cache_dir = Config.CACHE_DIR
        self.input_dir = Config.INPUT_DIR

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Configuration
        self.patch_size = Config.PATCH_SIZE
        if self.mode == "train":
            self.patches_per_image = Config.PATCHES_PER_IMAGE
            # Geometric Augmentations for Training to exploit noise symmetry
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ],
                additional_targets={"target": "image"},
            )
        else:
            self.patches_per_image = 1
            self.transform = None

        # Load Metadata
        if self.mode == "train":
            self.metadata = pd.read_csv(Config.TRAIN_METADATA_PATH)
        elif self.mode == "val":
            self.metadata = pd.read_csv(Config.VAL_METADATA_PATH)
        elif self.mode == "test":
            self.metadata = pd.read_csv(Config.TEST_METADATA_PATH)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Load Data into Memory
        # We process all images at init to ensure fast training loop
        self.data = []
        for _, row in self.metadata.iterrows():
            self.data.append(self.process_and_cache_image(row, self.load_cached_data))

    def process_and_cache_image(self, row, load_cached_data):
        """
        Loads, normalizes, and caches image data.
        Follows strict caching logic: Try Load -> If Fail -> Compute & Save.
        """
        img_id = str(row["id"])
        noisy_cache_path = os.path.join(self.cache_dir, f"{img_id}_noisy.npy")
        clean_cache_path = os.path.join(self.cache_dir, f"{img_id}_clean.npy")

        has_label = "label_path" in row and pd.notna(row["label_path"])

        # 1. Try to load from cache
        if load_cached_data:
            if os.path.exists(noisy_cache_path):
                # If we expect a label, check if it exists in cache too
                if has_label:
                    if os.path.exists(clean_cache_path):
                        noisy = np.load(noisy_cache_path)
                        clean = np.load(clean_cache_path)
                        return {"id": img_id, "noisy": noisy, "clean": clean}
                else:
                    # No label expected (Test set)
                    noisy = np.load(noisy_cache_path)
                    return {"id": img_id, "noisy": noisy, "clean": None}

        # 2. Compute from scratch

        # Load Noisy Image
        noisy_path = os.path.join(self.input_dir, row["feature_path"])
        if not os.path.exists(noisy_path):
            raise FileNotFoundError(f"Image not found: {noisy_path}")

        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_UNCHANGED)
        if noisy_img is None:
            raise ValueError(f"Failed to read image: {noisy_path}")

        # Handle Channels (Ensure Grayscale)
        if len(noisy_img.shape) == 3:
            noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)
        elif len(noisy_img.shape) == 4:  # RGBA
            noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGRA2GRAY)

        # Normalize to [0, 1] float32
        noisy_arr = noisy_img.astype(np.float32) / 255.0

        # Load Clean Image (if available)
        clean_arr = None
        if has_label:
            clean_path = os.path.join(self.input_dir, row["label_path"])
            if not os.path.exists(clean_path):
                raise FileNotFoundError(f"Image not found: {clean_path}")

            clean_img = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED)
            if clean_img is None:
                raise ValueError(f"Failed to read image: {clean_path}")

            if len(clean_img.shape) == 3:
                clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
            elif len(clean_img.shape) == 4:
                clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGRA2GRAY)

            clean_arr = clean_img.astype(np.float32) / 255.0

        # Save to Cache
        np.save(noisy_cache_path, noisy_arr)
        if clean_arr is not None:
            np.save(clean_cache_path, clean_arr)

        return {"id": img_id, "noisy": noisy_arr, "clean": clean_arr}

    def __len__(self):
        # For training, we artificially expand the dataset size to sample multiple patches
        return len(self.data) * self.patches_per_image

    def __getitem__(self, idx):
        # Map flat index to image index
        img_idx = idx // self.patches_per_image
        sample = self.data[img_idx]

        noisy = sample["noisy"]
        clean = sample["clean"]
        img_id = sample["id"]

        # Training: Random Patching and Augmentation
        if self.mode == "train":
            h, w = noisy.shape
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)

            # Pad if image is smaller than patch
            if pad_h > 0 or pad_w > 0:
                noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode="reflect")
                clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
                h, w = noisy.shape

            # Random Crop
            y = np.random.randint(0, h - self.patch_size + 1)
            x = np.random.randint(0, w - self.patch_size + 1)

            noisy_patch = noisy[y : y + self.patch_size, x : x + self.patch_size]
            clean_patch = clean[y : y + self.patch_size, x : x + self.patch_size]

            # Augmentation
            if self.transform:
                # Albumentations expects (H, W) or (H, W, C)
                # Our data is (H, W)
                augmented = self.transform(image=noisy_patch, target=clean_patch)
                noisy_patch = augmented["image"]
                clean_patch = augmented["target"]

            # Convert to Tensor (C, H, W)
            noisy_t = torch.from_numpy(noisy_patch).unsqueeze(0)
            clean_t = torch.from_numpy(clean_patch).unsqueeze(0)

            # Target is Noise Residual (Noisy - Clean)
            target_t = noisy_t - clean_t

            return noisy_t, target_t

        # Validation: Full Image
        elif self.mode == "val":
            # Convert to Tensor
            noisy_t = torch.from_numpy(noisy).unsqueeze(0)
            clean_t = torch.from_numpy(clean).unsqueeze(0)

            # Target is Noise Residual
            target_t = noisy_t - clean_t

            return noisy_t, target_t

        # Test: Full Image, No Target
        else:
            noisy_t = torch.from_numpy(noisy).unsqueeze(0)
            return noisy_t, img_id
