import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class TextDenoisingDataset(Dataset):
    """
    Dataset class for Text Denoising task.
    Supports High-Density Sampling for training and full-image loading for inference.
    """

    def __init__(self, metadata_path, mode="train", load_cached_data=True):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): Operation mode - 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npy files.
        """
        self.metadata_path = metadata_path
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Configuration
        self.patch_size = Config.PATCH_SIZE
        # For training, we artificially expand the dataset size to sample multiple patches per image per epoch
        self.patches_per_image = Config.PATCHES_PER_IMAGE if mode == "train" else 1

        # Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        self.df = pd.read_csv(metadata_path)

        # Load Data (with caching mechanism)
        self.data_cache = self._process_and_cache_data()

    def _process_and_cache_data(self):
        """
        Loads images into memory. Uses on-disk caching (.npy) to speed up subsequent runs.
        """
        data_list = []

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        for _, row in self.df.iterrows():
            img_id = str(row["id"])

            # Define cache paths
            noisy_cache_path = os.path.join(Config.CACHE_DIR, f"{img_id}_noisy.npy")
            clean_cache_path = os.path.join(Config.CACHE_DIR, f"{img_id}_clean.npy")

            item = {"id": img_id}

            # --- Load Noisy Image ---
            noisy_img = None
            # Try loading from cache
            if self.load_cached_data and os.path.exists(noisy_cache_path):
                try:
                    noisy_img = np.load(noisy_cache_path)
                except Exception:
                    noisy_img = None  # Force reload if cache is corrupt

            # Load from source if not in cache
            if noisy_img is None:
                feature_path = os.path.join(Config.INPUT_DIR, row["feature_path"])
                if not os.path.exists(feature_path):
                    continue  # Skip missing files

                # Read as grayscale
                img_raw = cv2.imread(feature_path, cv2.IMREAD_GRAYSCALE)
                if img_raw is None:
                    continue

                # Normalize to [0, 1] float32
                noisy_img = img_raw.astype(np.float32) / 255.0

                # Save to cache
                if self.load_cached_data:
                    np.save(noisy_cache_path, noisy_img)

            item["noisy"] = noisy_img

            # --- Load Clean Image (if available) ---
            if "label_path" in row and pd.notna(row["label_path"]):
                clean_img = None
                # Try loading from cache
                if self.load_cached_data and os.path.exists(clean_cache_path):
                    try:
                        clean_img = np.load(clean_cache_path)
                    except Exception:
                        clean_img = None

                # Load from source
                if clean_img is None:
                    label_path = os.path.join(Config.INPUT_DIR, row["label_path"])
                    if os.path.exists(label_path):
                        img_raw = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                        if img_raw is not None:
                            clean_img = img_raw.astype(np.float32) / 255.0
                            if self.load_cached_data:
                                np.save(clean_cache_path, clean_img)

                if clean_img is not None:
                    item["clean"] = clean_img

            data_list.append(item)

        return data_list

    def __len__(self):
        return len(self.data_cache) * self.patches_per_image

    def __getitem__(self, idx):
        # Map linear index to image index
        img_idx = idx // self.patches_per_image
        sample = self.data_cache[img_idx]

        noisy = sample["noisy"]
        clean = sample.get("clean")
        img_id = sample["id"]

        if self.mode == "train":
            # --- Training Mode: Random Crop & Augmentation ---
            h, w = noisy.shape

            # Pad if image is smaller than patch size (safety check)
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)

            if pad_h > 0 or pad_w > 0:
                noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode="reflect")
                if clean is not None:
                    clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
                h, w = noisy.shape

            # Random Crop
            y = np.random.randint(0, h - self.patch_size + 1)
            x = np.random.randint(0, w - self.patch_size + 1)

            patch_noisy = noisy[y : y + self.patch_size, x : x + self.patch_size]
            patch_clean = clean[y : y + self.patch_size, x : x + self.patch_size]

            # --- Augmentations ---
            # 1. Random Horizontal Flip
            if np.random.rand() > 0.5:
                patch_noisy = np.fliplr(patch_noisy)
                patch_clean = np.fliplr(patch_clean)

            # 2. Random Vertical Flip
            if np.random.rand() > 0.5:
                patch_noisy = np.flipud(patch_noisy)
                patch_clean = np.flipud(patch_clean)

            # 3. Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                patch_noisy = np.rot90(patch_noisy, k)
                patch_clean = np.rot90(patch_clean, k)

            # Convert to Tensor (C, H, W)
            # .copy() is required because flip/rot can create negative strides which torch doesn't support
            tensor_noisy = torch.from_numpy(patch_noisy.copy()).float().unsqueeze(0)
            tensor_clean = torch.from_numpy(patch_clean.copy()).float().unsqueeze(0)

            return tensor_noisy, tensor_clean

        else:
            # --- Val/Test Mode: Full Image ---
            tensor_noisy = torch.from_numpy(noisy).float().unsqueeze(0)

            if clean is not None:
                tensor_clean = torch.from_numpy(clean).float().unsqueeze(0)
                return tensor_noisy, tensor_clean, img_id
            else:
                return tensor_noisy, img_id
