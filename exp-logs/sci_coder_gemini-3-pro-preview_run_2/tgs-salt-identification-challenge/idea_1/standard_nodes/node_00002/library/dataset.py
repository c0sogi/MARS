import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A


class SaltDataset(Dataset):
    def __init__(self, metadata_path, config, mode="train", load_cached_data=True):
        """
        Dataset for Salt Segmentation.

        Args:
            metadata_path (str): Path to the metadata CSV file.
            config (Config): Configuration object containing paths and hyperparameters.
            mode (str): One of 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load pre-processed numpy arrays from disk.
        """
        self.config = config
        self.mode = mode
        self.metadata_path = metadata_path

        # Load Metadata (Lightweight)
        self.df = pd.read_csv(metadata_path)
        self.ids = self.df["id"].values

        # Process Depths (Vectorized)
        z_values = self.df["z"].values.astype(np.float32)
        self.depths = (z_values - self.config.DEPTH_MEAN) / self.config.DEPTH_STD

        # Setup Cache Paths
        cache_dir = os.path.join(self.config.WORKING_DIR, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        meta_name = os.path.basename(metadata_path).replace(".csv", "")
        self.cache_path_images = os.path.join(cache_dir, f"{meta_name}_images.npy")
        self.cache_path_masks = os.path.join(cache_dir, f"{meta_name}_masks.npy")

        # Load or Process Heavy Data (Images/Masks)
        self.images = None
        self.masks = None

        data_loaded = False
        if load_cached_data:
            data_loaded = self._try_load_cache()

        if not data_loaded:
            self._process_and_cache()

        # Define Augmentations
        self.transform = None
        if self.mode == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                    ),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                ]
            )

    def _try_load_cache(self):
        """Attempts to load images and masks from .npy files."""
        if not os.path.exists(self.cache_path_images):
            return False

        if self.mode != "test" and not os.path.exists(self.cache_path_masks):
            return False

        try:
            # Load images
            self.images = np.load(self.cache_path_images)

            # Load masks if not test
            if self.mode != "test":
                self.masks = np.load(self.cache_path_masks)

            # Verify lengths match metadata
            if len(self.images) != len(self.df):
                return False

            return True
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False

    def _process_and_cache(self):
        """Reads images from disk, applies padding/normalization, and saves to .npy."""
        n_samples = len(self.df)
        h, w = self.config.INPUT_SHAPE  # 128, 128
        orig_h, orig_w = self.config.ORIG_SHAPE

        # Calculate Padding
        pad_h = h - orig_h
        pad_w = w - orig_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Allocate Arrays
        self.images = np.zeros((n_samples, h, w), dtype=np.float32)
        if self.mode != "test":
            self.masks = np.zeros((n_samples, h, w), dtype=np.float32)

        input_root = self.config.INPUT_ROOT

        for idx, row in self.df.iterrows():
            # Process Image
            img_path = os.path.join(input_root, row["image_path"])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                img = np.zeros((orig_h, orig_w), dtype=np.uint8)

            img_padded = cv2.copyMakeBorder(
                img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
            )
            self.images[idx] = img_padded.astype(np.float32) / 255.0

            # Process Mask (Train/Val only)
            if self.mode != "test":
                mask_path = os.path.join(input_root, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if mask is not None:
                    mask_padded = cv2.copyMakeBorder(
                        mask,
                        pad_top,
                        pad_bottom,
                        pad_left,
                        pad_right,
                        cv2.BORDER_REFLECT,
                    )
                    # Binarize and Normalize
                    self.masks[idx] = (mask_padded > 127).astype(np.float32)
                else:
                    self.masks[idx] = np.zeros((h, w), dtype=np.float32)

        # Save to Cache
        np.save(self.cache_path_images, self.images)
        if self.mode != "test":
            np.save(self.cache_path_masks, self.masks)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve Data
        image = self.images[idx]
        depth = self.depths[idx]

        # Convert to Tensor (Add Channel Dimension)
        # For test set, we just need the tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, 128, 128)
        depth_tensor = torch.tensor([depth], dtype=torch.float32)

        if self.mode == "test":
            return image_tensor, depth_tensor, str(self.ids[idx])

        mask = self.masks[idx]

        # Apply Augmentations (Train Only)
        if self.mode == "train" and self.transform:
            # Albumentations expects HWC, but we have HW.
            # It works fine with HW for grayscale if passed correctly.
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to Tensor (Add Channel Dimension)
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, 128, 128)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, 128, 128)

        return image_tensor, depth_tensor, mask_tensor
