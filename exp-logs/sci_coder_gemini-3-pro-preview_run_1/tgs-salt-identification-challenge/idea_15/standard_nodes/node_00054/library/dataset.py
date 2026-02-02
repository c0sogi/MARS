import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class SaltDataset(Dataset):
    """
    Dataset class for Salt Segmentation.

    Features:
    - Loads images, masks, and depth information.
    - Implements caching mechanism to .npy files for fast loading.
    - Applies Reflection Padding to resize 101x101 -> 128x128.
    - Fuses depth information as a dense second channel.
    - Applies Horizontal Flip augmentation during training.
    """

    def __init__(self, mode="train", load_cached_data=True, transform=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.
            transform (callable, optional): Optional additional transforms (not used in main logic).
        """
        self.mode = mode
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Determine Metadata Source
        if self.mode == "train":
            self.csv_path = Config.TRAIN_CSV
        elif self.mode == "val":
            self.csv_path = Config.VAL_CSV
        else:
            self.csv_path = Config.TEST_CSV

        # Setup Cache Directory
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define Cache Paths
        self.cache_paths = {
            "images": os.path.join(self.cache_dir, f"{mode}_images.npy"),
            "masks": os.path.join(self.cache_dir, f"{mode}_masks.npy"),
            "depths": os.path.join(self.cache_dir, f"{mode}_depths.npy"),
            "ids": os.path.join(self.cache_dir, f"{mode}_ids.npy"),
        }

        # Load Data (Cache or Source)
        self._load_data()

        # Precompute Padding Parameters (101 -> 128)
        self.pad_h = Config.IMG_SIZE - Config.ORIG_SIZE
        self.pad_w = Config.IMG_SIZE - Config.ORIG_SIZE
        self.pad_top = self.pad_h // 2
        self.pad_bottom = self.pad_h - self.pad_top
        self.pad_left = self.pad_w // 2
        self.pad_right = self.pad_w - self.pad_left

        # Calculate Depth Statistics for Normalization
        # We normalize depth to [0, 1] based on the range in the current split
        self.depth_min = self.depths.min()
        self.depth_max = self.depths.max()
        if self.depth_max == self.depth_min:
            self.depth_max += 1.0  # Avoid division by zero

    def _load_data(self):
        """
        Loads data from .npy cache if available and requested.
        Otherwise, reads from CSV/Images, processes, and saves to cache.
        """
        cache_exists = all(os.path.exists(p) for p in self.cache_paths.values())

        if self.load_cached_data and cache_exists:
            try:
                self.images = np.load(self.cache_paths["images"])
                self.masks = np.load(self.cache_paths["masks"])
                self.depths = np.load(self.cache_paths["depths"])
                self.ids = np.load(self.cache_paths["ids"])
                return
            except Exception as e:
                print(f"Cache load failed ({e}). Re-processing from source.")

        # Load from Source
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        images_list = []
        masks_list = []
        depths_list = []
        ids_list = []

        for idx, row in df.iterrows():
            # Load Image
            img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Load Mask
            if self.mode in ["train", "val"]:
                mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # Ensure binary 0/1
                mask = (mask > 127).astype(np.uint8)
            else:
                # Placeholder for test
                mask = np.zeros_like(img, dtype=np.uint8)

            images_list.append(img)
            masks_list.append(mask)
            depths_list.append(row["z"])
            ids_list.append(row["id"])

        # Convert to Numpy Arrays
        self.images = np.array(images_list, dtype=np.uint8)
        self.masks = np.array(masks_list, dtype=np.uint8)
        self.depths = np.array(depths_list, dtype=np.float32)
        self.ids = np.array(ids_list)

        # Save to Cache
        np.save(self.cache_paths["images"], self.images)
        np.save(self.cache_paths["masks"], self.masks)
        np.save(self.cache_paths["depths"], self.depths)
        np.save(self.cache_paths["ids"], self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 1. Retrieve Raw Data
        img = self.images[idx]  # (101, 101) uint8
        mask = self.masks[idx]  # (101, 101) uint8
        z = self.depths[idx]  # scalar float
        img_id = self.ids[idx]

        # 2. Normalize Image (0-1)
        img = img.astype(np.float32) / 255.0

        # 3. Normalize Depth (0-1)
        z_norm = (z - self.depth_min) / (self.depth_max - self.depth_min)

        # 4. Reflection Padding (101x101 -> 128x128)
        img_padded = cv2.copyMakeBorder(
            img,
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
            cv2.BORDER_REFLECT_101,
        )

        mask_padded = cv2.copyMakeBorder(
            mask,
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
            cv2.BORDER_REFLECT_101,
        )

        # 5. Augmentation (Horizontal Flip) - Train Only
        if self.mode == "train":
            if np.random.rand() > 0.5:
                # Use ascontiguousarray to prevent negative stride issues in torch
                img_padded = np.ascontiguousarray(np.fliplr(img_padded))
                mask_padded = np.ascontiguousarray(np.fliplr(mask_padded))

        # 6. Fuse Depth Channel
        # Create a dense depth channel matching the image spatial dimensions
        depth_channel = np.full_like(img_padded, z_norm, dtype=np.float32)

        # 7. Stack Channels -> (2, 128, 128)
        # Channel 0: Image, Channel 1: Depth
        img_final = np.stack([img_padded, depth_channel], axis=0)

        # 8. Format Mask -> (1, 128, 128)
        mask_final = mask_padded[np.newaxis, :, :].astype(np.float32)

        # 9. Convert to Tensor
        return torch.from_numpy(img_final), torch.from_numpy(mask_final), img_id
