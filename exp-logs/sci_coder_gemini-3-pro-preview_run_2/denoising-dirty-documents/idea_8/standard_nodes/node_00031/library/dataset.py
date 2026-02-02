import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class DenoisingDataset(Dataset):
    def __init__(self, csv_path, mode="train", load_cached_data=True):
        """
        Dataset for image denoising with caching and high-density sampling.

        Args:
            csv_path (str): Path to the metadata CSV file.
            mode (str): Operation mode ('train', 'val', 'test').
            load_cached_data (bool): Whether to use existing cached .npy files.
        """
        self.mode = mode
        self.df = pd.read_csv(csv_path)
        self.data = []
        self.patch_size = Config.PATCH_SIZE

        # In training, we sample multiple patches per image per epoch
        self.patches_per_image = Config.PATCHES_PER_IMAGE if mode == "train" else 1

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Load all data into RAM
        self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        """
        Iterates through metadata, loading images from cache or processing raw files.
        """
        for _, row in self.df.iterrows():
            image_id = str(row["id"])

            # Define cache filenames
            noisy_cache_path = os.path.join(Config.CACHE_DIR, f"{image_id}_noisy.npy")
            clean_cache_path = os.path.join(Config.CACHE_DIR, f"{image_id}_clean.npy")

            # Get relative paths from CSV
            noisy_rel_path = row["feature_path"]
            clean_rel_path = row.get("label_path", None)

            # Load Noisy Image
            noisy_img = self._get_image(
                noisy_rel_path, noisy_cache_path, load_cached_data
            )

            # Load Clean Image (if available)
            clean_img = None
            if clean_rel_path:
                clean_img = self._get_image(
                    clean_rel_path, clean_cache_path, load_cached_data
                )

            self.data.append({"id": image_id, "noisy": noisy_img, "clean": clean_img})

    def _get_image(self, rel_path, cache_path, load_cached_data):
        """
        Retrieves an image from cache or processes it from source.
        """
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                # If load fails, proceed to process from source
                pass

        # 2. Process from source
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")

        # Read image (preserve channels if any, though likely grayscale)
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image with cv2: {full_path}")

        # Ensure correct shape (H, W, C)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # Convert BGR to Gray if necessary (task specifies grayscale intensity)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)

        # Normalize to [0, 1] float32
        img = img.astype(np.float32) / 255.0

        # Save to cache
        np.save(cache_path, img)

        return img

    def __len__(self):
        """
        Returns total number of samples.
        For training, this is num_images * patches_per_image.
        """
        return len(self.data) * self.patches_per_image

    def __getitem__(self, idx):
        """
        Returns a sample.
        Train: Random 128x128 patch with augmentations.
        Val/Test: Full image.
        """
        # Map linear index to image index
        img_idx = idx // self.patches_per_image
        sample = self.data[img_idx]

        noisy = sample["noisy"]
        clean = sample["clean"]
        img_id = sample["id"]

        if self.mode == "train":
            # --- High-Density Sampling & Augmentation ---
            h, w, c = noisy.shape

            # Random Crop
            # Ensure image is larger than patch size (EDA confirms min dims > 128)
            max_y = h - self.patch_size
            max_x = w - self.patch_size

            y = np.random.randint(0, max_y + 1)
            x = np.random.randint(0, max_x + 1)

            patch_noisy = noisy[y : y + self.patch_size, x : x + self.patch_size, :]
            patch_clean = clean[y : y + self.patch_size, x : x + self.patch_size, :]

            # Geometric Augmentations
            # 1. Random Flips
            if np.random.rand() > 0.5:
                patch_noisy = np.flipud(patch_noisy)
                patch_clean = np.flipud(patch_clean)
            if np.random.rand() > 0.5:
                patch_noisy = np.fliplr(patch_noisy)
                patch_clean = np.fliplr(patch_clean)

            # 2. Random 90-degree Rotations
            k = np.random.randint(0, 4)
            if k > 0:
                patch_noisy = np.rot90(patch_noisy, k)
                patch_clean = np.rot90(patch_clean, k)

            # Convert to Tensor (C, H, W)
            # ascontiguousarray is needed because numpy flips/rotates can create negative strides
            tensor_noisy = torch.from_numpy(np.ascontiguousarray(patch_noisy)).permute(
                2, 0, 1
            )
            tensor_clean = torch.from_numpy(np.ascontiguousarray(patch_clean)).permute(
                2, 0, 1
            )

            return tensor_noisy, tensor_clean

        elif self.mode == "val":
            # --- Validation: Full Images ---
            tensor_noisy = torch.from_numpy(noisy).permute(2, 0, 1)
            tensor_clean = torch.from_numpy(clean).permute(2, 0, 1)
            return tensor_noisy, tensor_clean, img_id

        else:
            # --- Test: Full Noisy Image Only ---
            tensor_noisy = torch.from_numpy(noisy).permute(2, 0, 1)
            return tensor_noisy, img_id
