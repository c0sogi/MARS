import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    PATCH_SIZE,
    SAMPLES_PER_IMAGE,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
)


class DenoisingDataset(Dataset):
    """
    Dataset class for the Denoising Task.
    Handles loading, caching, normalizing, cropping, and augmenting data.
    """

    def __init__(
        self,
        mode="train",
        csv_path=None,
        load_cached_data=True,
        samples_per_image=SAMPLES_PER_IMAGE,
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            csv_path (str): Path to the metadata CSV file.
            load_cached_data (bool): Whether to load/save data from/to the cache directory.
            samples_per_image (int): Number of random patches to extract per image per epoch (train only).
        """
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.samples_per_image = samples_per_image
        self.patch_size = PATCH_SIZE

        # Determine CSV path if not provided
        if csv_path is None:
            if mode == "train":
                csv_path = TRAIN_CSV
            elif mode == "val":
                csv_path = VAL_CSV
            elif mode == "test":
                csv_path = TEST_CSV
            else:
                raise ValueError(f"Unknown mode: {mode}")

        # Load Metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        self.metadata = pd.read_csv(csv_path)

        # Preload data into memory
        self.data = []
        self._load_dataset()

    def _load_dataset(self):
        """
        Iterates through metadata and loads images into memory.
        Implements caching logic.
        """
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        for _, row in self.metadata.iterrows():
            img_id = str(row["id"])

            # Load Noisy Image (Feature)
            feature_rel_path = row["feature_path"]
            noisy_img = self._load_single_image(feature_rel_path, img_id, "noisy")

            if self.mode in ["train", "val"]:
                # Load Clean Image (Label)
                label_rel_path = row["label_path"]
                clean_img = self._load_single_image(label_rel_path, img_id, "clean")
                self.data.append({"id": img_id, "noisy": noisy_img, "clean": clean_img})
            else:
                # Test mode (no label)
                self.data.append({"id": img_id, "noisy": noisy_img})

    def _load_single_image(self, rel_path, img_id, suffix):
        """
        Loads a single image, handling caching logic.
        Returns a float32 numpy array normalized to [0, 1].
        """
        cache_filename = f"{img_id}_{suffix}.npy"
        cache_path = os.path.join(CACHE_DIR, cache_filename)

        # 1. Try to load from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                # If load fails, fall back to processing from scratch
                pass

        # 2. Process from scratch
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")

        # Read as grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {full_path}")

        # Normalize to [0, 1] and convert to float32
        img = img.astype(np.float32) / 255.0

        # 3. Save to cache
        np.save(cache_path, img)

        return img

    def __len__(self):
        if self.mode == "train":
            # Virtual length for high-density sampling
            return len(self.data) * self.samples_per_image
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "train":
            # Map virtual index to actual image index
            img_idx = idx // self.samples_per_image
            sample = self.data[img_idx]

            noisy = sample["noisy"]
            clean = sample["clean"]

            h, w = noisy.shape

            # Random Crop
            # Ensure image is larger than patch size
            if h < self.patch_size or w < self.patch_size:
                # Fallback: Resize or Pad if strictly necessary,
                # but EDA showed min dims > patch_size.
                pad_h = max(0, self.patch_size - h)
                pad_w = max(0, self.patch_size - w)
                if pad_h > 0 or pad_w > 0:
                    noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode="reflect")
                    clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
                    h, w = noisy.shape

            y = np.random.randint(0, h - self.patch_size + 1)
            x = np.random.randint(0, w - self.patch_size + 1)

            noisy_patch = noisy[y : y + self.patch_size, x : x + self.patch_size]
            clean_patch = clean[y : y + self.patch_size, x : x + self.patch_size]

            # Geometric Augmentations
            # 1. Random Flip
            if np.random.rand() > 0.5:
                # Horizontal
                noisy_patch = np.fliplr(noisy_patch)
                clean_patch = np.fliplr(clean_patch)

            if np.random.rand() > 0.5:
                # Vertical
                noisy_patch = np.flipud(noisy_patch)
                clean_patch = np.flipud(clean_patch)

            # 2. Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                noisy_patch = np.rot90(noisy_patch, k)
                clean_patch = np.rot90(clean_patch, k)

            # Convert to Tensor (C, H, W)
            # Arrays are currently (H, W), need to add channel dim
            noisy_t = torch.from_numpy(noisy_patch.copy()).unsqueeze(0)
            clean_t = torch.from_numpy(clean_patch.copy()).unsqueeze(0)

            return noisy_t, clean_t

        elif self.mode == "val":
            sample = self.data[idx]
            noisy = sample["noisy"]
            clean = sample["clean"]
            img_id = sample["id"]

            # Return full images
            noisy_t = torch.from_numpy(noisy).unsqueeze(0)
            clean_t = torch.from_numpy(clean).unsqueeze(0)

            return noisy_t, clean_t, img_id

        else:  # test
            sample = self.data[idx]
            noisy = sample["noisy"]
            img_id = sample["id"]

            noisy_t = torch.from_numpy(noisy).unsqueeze(0)

            return noisy_t, img_id
