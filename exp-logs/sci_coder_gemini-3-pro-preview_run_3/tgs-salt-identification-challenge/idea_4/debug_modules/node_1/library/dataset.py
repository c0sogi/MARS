import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from library.config import Config


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles loading, caching, padding, depth fusion, and augmentation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mode: str,
        config: Config,
        load_cached_data: bool = True,
        limit_size: int = None,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (paths, ids, depths, etc.).
            mode (str): 'train', 'val', or 'test'.
            config (Config): Configuration object.
            load_cached_data (bool): Whether to load data from numpy cache if available.
            limit_size (int, optional): Limit dataset size for debugging.
        """
        self.mode = mode
        self.config = config
        self.df = df

        # Apply limit for debugging if specified
        if limit_size is not None:
            self.df = self.df.iloc[:limit_size].reset_index(drop=True)

        # Cache filenames
        debug_suffix = "_debug" if config.DEBUG else ""
        self.cache_images_path = os.path.join(
            config.CACHE_DIR, f"cached_{mode}{debug_suffix}_images.npy"
        )
        self.cache_masks_path = os.path.join(
            config.CACHE_DIR, f"cached_{mode}{debug_suffix}_masks.npy"
        )
        self.cache_depths_path = os.path.join(
            config.CACHE_DIR, f"cached_{mode}{debug_suffix}_depths.npy"
        )
        self.cache_ids_path = os.path.join(
            config.CACHE_DIR, f"cached_{mode}{debug_suffix}_ids.npy"
        )

        # Load or Process Data
        self._load_data(load_cached_data)

        # Calculate Depth Statistics for Normalization
        # We use the min/max of the current split.
        # In a strict setting, we might use global stats, but per-split is robust enough for this task.
        self.depth_min = self.depths.min()
        self.depth_max = self.depths.max()

        # Define Augmentations
        # Conservative: Flip and Brightness only. No geometric distortions.
        if self.mode == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                ]
            )
        else:
            self.transform = None

    def _load_data(self, load_cached_data):
        """
        Loads data from cache or processes from scratch and saves to cache.
        """
        # Ensure cache directory exists
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)

        # Check if all cache files exist
        cache_exists = (
            os.path.exists(self.cache_images_path)
            and os.path.exists(self.cache_depths_path)
            and os.path.exists(self.cache_ids_path)
        )

        # For train/val, we also need masks
        if self.mode != "test":
            cache_exists = cache_exists and os.path.exists(self.cache_masks_path)

        if load_cached_data and cache_exists:
            # print(f"Loading {self.mode} data from cache...")
            self.images = np.load(self.cache_images_path)
            self.depths = np.load(self.cache_depths_path)
            self.ids = np.load(self.cache_ids_path, allow_pickle=True)
            if self.mode != "test":
                self.masks = np.load(self.cache_masks_path)
            else:
                self.masks = None
        else:
            # print(f"Processing {self.mode} data from scratch...")
            self._process_and_cache()

    def _process_and_cache(self):
        """
        Reads images from disk, stacks them into numpy arrays, and saves to cache.
        """
        images_list = []
        masks_list = []
        depths_list = []
        ids_list = []

        for _, row in self.df.iterrows():
            img_id = row["id"]

            # Load Image (RGB)
            img_path = os.path.join(self.config.INPUT_DIR, row["image_path"])
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images_list.append(img)

            # Load Mask (if available)
            if self.mode != "test":
                mask_path = os.path.join(self.config.INPUT_DIR, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                # Expand dims to (H, W, 1)
                mask = np.expand_dims(mask, axis=-1)
                masks_list.append(mask)

            # Load Depth
            depths_list.append(row["z"])
            ids_list.append(img_id)

        # Convert to Numpy Arrays
        self.images = np.array(images_list, dtype=np.uint8)
        self.depths = np.array(depths_list, dtype=np.float32)
        self.ids = np.array(ids_list)

        # Save to Cache
        np.save(self.cache_images_path, self.images)
        np.save(self.cache_depths_path, self.depths)
        np.save(self.cache_ids_path, self.ids)

        if self.mode != "test":
            self.masks = np.array(masks_list, dtype=np.uint8)
            np.save(self.cache_masks_path, self.masks)
        else:
            self.masks = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Retrieve Data
        image = self.images[idx]  # (101, 101, 3)
        depth = self.depths[idx]
        img_id = self.ids[idx]

        if self.mode != "test":
            mask = self.masks[idx]  # (101, 101, 1)
        else:
            # Create dummy mask for test set
            mask = np.zeros(
                (self.config.IMG_SIZE_ORIG, self.config.IMG_SIZE_ORIG, 1),
                dtype=np.uint8,
            )

        # 2. Padding (Reflection)
        # Pad from 101x101 to 128x128
        # Calculate padding amounts
        h, w = image.shape[:2]
        target_h, target_w = self.config.IMG_SIZE_MODEL, self.config.IMG_SIZE_MODEL

        pad_h = target_h - h
        pad_w = target_w - w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply reflection padding
        # cv2.copyMakeBorder expects BGR/RGB images or Grayscale
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
        mask = cv2.copyMakeBorder(
            mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )

        # Mask becomes (128, 128) after padding if it was (101, 101, 1) passed to cv2
        # Ensure mask is (H, W, 1)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)

        # 3. Augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 4. Normalization & Depth Fusion
        # Normalize Image (0-1)
        image = image.astype(np.float32) / 255.0

        # Normalize Depth (0-1)
        # Avoid division by zero if min == max (unlikely but safe)
        if self.depth_max > self.depth_min:
            d_norm = (depth - self.depth_min) / (self.depth_max - self.depth_min)
        else:
            d_norm = 0.0

        # Create Depth Channel
        # Shape (128, 128, 1)
        depth_channel = np.full(
            (self.config.IMG_SIZE_MODEL, self.config.IMG_SIZE_MODEL, 1),
            d_norm,
            dtype=np.float32,
        )

        # Concatenate to create 4-channel input
        # (128, 128, 3) + (128, 128, 1) -> (128, 128, 4)
        image_fused = np.concatenate([image, depth_channel], axis=-1)

        # 5. To Tensor (CHW)
        # (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image_fused).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float()

        # For test mode, we don't strictly need the mask, but returning a dummy one keeps signature consistent
        return image_tensor, mask_tensor, img_id
