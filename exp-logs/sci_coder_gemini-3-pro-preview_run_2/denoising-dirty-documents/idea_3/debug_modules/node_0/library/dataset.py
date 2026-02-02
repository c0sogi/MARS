import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class DenoisingDataset(Dataset):
    def __init__(
        self,
        metadata_file: str,
        mode: str = "train",
        patches_per_image: int = Config.PATCHES_PER_IMAGE,
        load_cached_data: bool = Config.LOAD_CACHED_DATA,
    ):
        """
        Dataset for Denoising Task with High-Density Sampling and Caching.

        Args:
            metadata_file (str): Path to the metadata CSV file.
            mode (str): Operation mode - 'train', 'val', or 'test'.
            patches_per_image (int): Number of random patches to sample per image (train mode).
            load_cached_data (bool): Whether to attempt loading from numpy cache.
        """
        self.mode = mode
        # Apply high-density sampling strategy only for training
        self.patches_per_image = patches_per_image if mode == "train" else 1
        self.patch_size = Config.PATCH_SIZE

        # Load Metadata
        self.metadata = pd.read_csv(metadata_file)

        # Setup Cache Directory
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Data Container
        self.data = []

        # Execute Caching and Loading Logic
        self._load_and_cache_data(load_cached_data)

    def _load_and_cache_data(self, load_cached_data: bool):
        """
        Loads images, normalizing them and caching them as .npy files.
        Strictly follows the logic: Check Cache -> (If miss) Load Raw -> Save Cache.
        """
        for _, row in self.metadata.iterrows():
            img_id = str(row["id"])

            # Define Cache Paths
            noisy_cache_path = os.path.join(self.cache_dir, f"{img_id}_noisy.npy")
            clean_cache_path = os.path.join(self.cache_dir, f"{img_id}_clean.npy")

            noisy_img = None
            clean_img = None

            # 1. Attempt to Load from Cache
            if load_cached_data:
                if os.path.exists(noisy_cache_path):
                    try:
                        noisy_img = np.load(noisy_cache_path)
                        # For train/val, we also need the clean image
                        if self.mode != "test":
                            if os.path.exists(clean_cache_path):
                                clean_img = np.load(clean_cache_path)
                            else:
                                # Partial cache miss (clean missing), force reload both
                                noisy_img = None
                    except Exception:
                        # Corrupt file, force reload
                        noisy_img = None

            # 2. Process from Scratch if Cache Miss or Forced
            if noisy_img is None:
                # Load Noisy Image
                f_path = os.path.join(Config.INPUT_DIR, row["feature_path"])
                # Read as grayscale (single channel)
                n_img_raw = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)

                if n_img_raw is None:
                    continue  # Skip broken file paths

                # Normalize to [0, 1] float32
                noisy_img = n_img_raw.astype(np.float32) / 255.0

                # Save to Cache
                np.save(noisy_cache_path, noisy_img)

                # Load Clean Image (if available/needed)
                if self.mode != "test":
                    l_path = os.path.join(Config.INPUT_DIR, row["label_path"])
                    c_img_raw = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)

                    if c_img_raw is None:
                        continue

                    # Normalize
                    clean_img = c_img_raw.astype(np.float32) / 255.0

                    # Save to Cache
                    np.save(clean_cache_path, clean_img)

            # Store in memory
            sample = {"id": img_id, "noisy": noisy_img}
            if clean_img is not None:
                sample["clean"] = clean_img

            self.data.append(sample)

    def __len__(self):
        """
        Returns the total number of samples.
        For training, this is num_images * patches_per_image.
        """
        return len(self.data) * self.patches_per_image

    def __getitem__(self, idx):
        """
        Retrieves a sample.
        Train: Random Crop + Augmentation -> (Noisy Patch, Residual Patch)
        Val: Center Crop -> (Noisy Patch, Residual Patch)
        Test: Full Image -> (Noisy Image, ID)
        """
        # Map flat index to image index
        img_idx = idx // self.patches_per_image
        sample = self.data[img_idx]

        noisy = sample["noisy"]

        # --- TEST MODE ---
        if self.mode == "test":
            # Return full image for inference (tiling handled by inference loop)
            # Shape: (1, H, W)
            noisy_t = torch.from_numpy(noisy).unsqueeze(0)
            return noisy_t, sample["id"]

        clean = sample["clean"]
        h, w = noisy.shape

        # --- TRAIN MODE ---
        if self.mode == "train":
            # Random Crop
            # np.random.randint is exclusive on high, so +1 is needed for inclusive range logic
            y_max = h - self.patch_size + 1
            x_max = w - self.patch_size + 1

            # Safety check if image is smaller than patch (unlikely given EDA)
            y = np.random.randint(0, max(1, y_max))
            x = np.random.randint(0, max(1, x_max))

            n_patch = noisy[y : y + self.patch_size, x : x + self.patch_size]
            c_patch = clean[y : y + self.patch_size, x : x + self.patch_size]

            # Geometric Augmentation
            # 1. Random Flip (Horizontal/Vertical)
            if np.random.rand() > 0.5:
                n_patch = np.flipud(n_patch)
                c_patch = np.flipud(c_patch)
            if np.random.rand() > 0.5:
                n_patch = np.fliplr(n_patch)
                c_patch = np.fliplr(c_patch)

            # 2. Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                n_patch = np.rot90(n_patch, k)
                c_patch = np.rot90(c_patch, k)

        # --- VAL MODE ---
        else:
            # Deterministic Center Crop
            y = (h - self.patch_size) // 2
            x = (w - self.patch_size) // 2
            n_patch = noisy[y : y + self.patch_size, x : x + self.patch_size]
            c_patch = clean[y : y + self.patch_size, x : x + self.patch_size]

        # Fix negative strides caused by numpy flips/rotates before converting to Tensor
        n_patch = n_patch.copy()
        c_patch = c_patch.copy()

        # Compute Target: Noise Residual
        # Model predicts Noise, so Target = Noisy - Clean
        residual = n_patch - c_patch

        # Convert to Tensor (C, H, W)
        n_t = torch.from_numpy(n_patch).unsqueeze(0)
        r_t = torch.from_numpy(residual).unsqueeze(0)

        return n_t, r_t
