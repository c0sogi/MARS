import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class DenoisingDataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True):
        """
        Dataset for Denoising Task.

        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load pre-processed .npy files.
                                     If False or file missing, processes from scratch and saves.
        """
        self.mode = mode
        self.patch_size = Config.PATCH_SIZE
        self.patches_per_image = Config.PATCHES_PER_IMAGE

        # Determine paths based on mode
        self.base_cache_dir = os.path.join(Config.WORKING_DIR, "cache")

        if self.mode == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
            self.cache_dir = os.path.join(self.base_cache_dir, "train")
        elif self.mode == "val":
            self.metadata_path = Config.VAL_METADATA_PATH
            self.cache_dir = os.path.join(self.base_cache_dir, "val")
        elif self.mode == "test":
            self.metadata_path = Config.TEST_METADATA_PATH
            self.cache_dir = os.path.join(self.base_cache_dir, "test")
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Load Metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        self.metadata = pd.read_csv(self.metadata_path)

        # Load and Cache Data
        self.data = self._load_and_cache_images(load_cached_data)

    def _load_and_cache_images(self, load_cached_data):
        """
        Loads images, processing them from scratch or loading from .npy cache.
        """
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        images_data = []

        for _, row in self.metadata.iterrows():
            img_id = str(row["id"])

            # Define cache filenames
            noisy_cache_path = os.path.join(self.cache_dir, f"{img_id}_noisy.npy")
            clean_cache_path = os.path.join(self.cache_dir, f"{img_id}_clean.npy")

            has_label = "label_path" in row and pd.notna(row["label_path"])

            # 1. Try Loading from Cache
            loaded = False
            noisy_img = None
            clean_img = None

            if load_cached_data:
                if os.path.exists(noisy_cache_path):
                    if has_label:
                        if os.path.exists(clean_cache_path):
                            noisy_img = np.load(noisy_cache_path)
                            clean_img = np.load(clean_cache_path)
                            loaded = True
                    else:
                        noisy_img = np.load(noisy_cache_path)
                        loaded = True

            # 2. Process from Scratch if needed
            if not loaded:
                # Load Noisy Image
                feat_path = os.path.join(Config.INPUT_DIR, row["feature_path"])
                raw_noisy = cv2.imread(feat_path, cv2.IMREAD_GRAYSCALE)
                if raw_noisy is None:
                    raise FileNotFoundError(f"Could not read image at {feat_path}")

                # Normalize [0, 1]
                noisy_img = raw_noisy.astype(np.float32) / 255.0
                np.save(noisy_cache_path, noisy_img)

                # Load Clean Image (if available)
                if has_label:
                    lbl_path = os.path.join(Config.INPUT_DIR, row["label_path"])
                    raw_clean = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
                    if raw_clean is None:
                        raise FileNotFoundError(f"Could not read image at {lbl_path}")

                    clean_img = raw_clean.astype(np.float32) / 255.0
                    np.save(clean_cache_path, clean_img)

            images_data.append({"id": img_id, "noisy": noisy_img, "clean": clean_img})

        return images_data

    def __len__(self):
        # High-density sampling for training
        if self.mode == "train":
            return len(self.data) * self.patches_per_image
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "train":
            # Map linear index to image index
            img_idx = idx // self.patches_per_image
            sample = self.data[img_idx]

            noisy = sample["noisy"]
            clean = sample["clean"]

            h, w = noisy.shape

            # Handle cases where image is smaller than patch size (padding)
            if h < self.patch_size or w < self.patch_size:
                pad_h = max(0, self.patch_size - h)
                pad_w = max(0, self.patch_size - w)
                noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode="reflect")
                clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
                h, w = noisy.shape

            # Random Crop
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)

            noisy_patch = noisy[
                top : top + self.patch_size, left : left + self.patch_size
            ]
            clean_patch = clean[
                top : top + self.patch_size, left : left + self.patch_size
            ]

            # Geometric Augmentations
            # 1. Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            noisy_patch = np.rot90(noisy_patch, k)
            clean_patch = np.rot90(clean_patch, k)

            # 2. Random Vertical Flip
            if np.random.random() > 0.5:
                noisy_patch = np.flipud(noisy_patch)
                clean_patch = np.flipud(clean_patch)

            # 3. Random Horizontal Flip
            if np.random.random() > 0.5:
                noisy_patch = np.fliplr(noisy_patch)
                clean_patch = np.fliplr(clean_patch)

            # Convert to Tensor [C, H, W]
            # .copy() is required because numpy flips/rotates return negative strides
            noisy_t = torch.from_numpy(noisy_patch.copy()).unsqueeze(0).float()
            clean_t = torch.from_numpy(clean_patch.copy()).unsqueeze(0).float()

            return noisy_t, clean_t

        elif self.mode == "val":
            sample = self.data[idx]
            noisy = sample["noisy"]
            clean = sample["clean"]
            img_id = sample["id"]

            noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
            clean_t = torch.from_numpy(clean).unsqueeze(0).float()

            return noisy_t, clean_t, img_id

        elif self.mode == "test":
            sample = self.data[idx]
            noisy = sample["noisy"]
            img_id = sample["id"]

            noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()

            return noisy_t, img_id
