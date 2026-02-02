import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(42)


class SaltDataset(Dataset):
    def __init__(
        self,
        mode="train",
        root_dir="./input",
        work_dir="./working/idea_9",
        load_cached_data=True,
    ):
        """
        Dataset for Salt Segmentation.

        Args:
            mode (str): 'train', 'val', or 'test'.
            root_dir (str): Path to input directory.
            work_dir (str): Path to working directory for caching.
            load_cached_data (bool): Whether to use cached .npy files.
        """
        self.mode = mode
        self.root_dir = root_dir
        self.work_dir = work_dir
        self.load_cached_data = load_cached_data

        # Metadata paths
        self.meta_files = {
            "train": "./metadata/train.csv",
            "val": "./metadata/val.csv",
            "test": "./metadata/test.csv",
        }

        if mode not in self.meta_files:
            raise ValueError(
                f"Invalid mode {mode}. Expected one of {list(self.meta_files.keys())}"
            )

        self.meta_path = self.meta_files[mode]
        self.df = pd.read_csv(self.meta_path)

        # Create work dir if not exists
        os.makedirs(self.work_dir, exist_ok=True)

        # Load data (cached or fresh)
        self.images, self.masks, self.depths, self.ids = self._load_data()

    def _load_data(self):
        """
        Loads data from cache or processes from scratch.

        Returns:
            images (np.array): (N, 101, 101) uint8
            masks (np.array): (N, 101, 101) uint8 (or None for test)
            depths (np.array): (N,) float
            ids (np.array): (N,) string
        """
        cache_prefix = f"{self.mode}"
        img_cache = os.path.join(self.work_dir, f"{cache_prefix}_images.npy")
        mask_cache = os.path.join(self.work_dir, f"{cache_prefix}_masks.npy")
        depth_cache = os.path.join(self.work_dir, f"{cache_prefix}_depths.npy")
        id_cache = os.path.join(self.work_dir, f"{cache_prefix}_ids.npy")

        # Check if all required cache files exist
        files_exist = (
            os.path.exists(img_cache)
            and os.path.exists(depth_cache)
            and os.path.exists(id_cache)
        )

        if self.mode != "test":
            files_exist = files_exist and os.path.exists(mask_cache)

        if self.load_cached_data and files_exist:
            print(f"Loading cached data for {self.mode} from {self.work_dir}...")
            images = np.load(img_cache)
            depths = np.load(depth_cache)
            ids = np.load(id_cache)
            masks = np.load(mask_cache) if self.mode != "test" else None
            return images, masks, depths, ids

        print(f"Processing data for {self.mode}...")

        images = []
        masks = []
        depths = []
        ids = []

        for idx, row in self.df.iterrows():
            # Load ID and Depth
            img_id = str(row["id"])
            depth = float(row["z"])

            # Load Image (Grayscale)
            # Metadata paths are relative, e.g., "train/images/xxxx.png"
            img_path = os.path.join(self.root_dir, row["image_path"])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                # Fallback for safety, though validation ensures paths exist
                img = np.zeros((101, 101), dtype=np.uint8)

            images.append(img)
            depths.append(depth)
            ids.append(img_id)

            # Load Mask (if not test)
            if self.mode != "test":
                mask_path = os.path.join(self.root_dir, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.zeros((101, 101), dtype=np.uint8)
                # Binarize mask (0 or 255 -> 0 or 1)
                mask = (mask > 127).astype(np.uint8)
                masks.append(mask)

        # Convert to numpy arrays
        images = np.array(images, dtype=np.uint8)
        depths = np.array(depths, dtype=np.float32)
        ids = np.array(ids)

        if self.mode != "test":
            masks = np.array(masks, dtype=np.uint8)
        else:
            masks = None

        # Save to cache
        print(f"Saving cached data for {self.mode} to {self.work_dir}...")
        np.save(img_cache, images)
        np.save(depth_cache, depths)
        np.save(id_cache, ids)
        if masks is not None:
            np.save(mask_cache, masks)

        return images, masks, depths, ids

    def _reflection_pad(self, img, target_size=128):
        """
        Pads image from 101x101 to target_size x target_size using reflection.
        """
        h, w = img.shape[:2]
        pad_h = target_size - h
        pad_w = target_size - w

        if pad_h < 0 or pad_w < 0:
            return img

        p_l = pad_w // 2
        p_r = pad_w - p_l
        p_t = pad_h // 2
        p_b = pad_h - p_t

        # BORDER_REFLECT_101 corresponds to PyTorch's 'reflect' padding
        padded = cv2.copyMakeBorder(img, p_t, p_b, p_l, p_r, cv2.BORDER_REFLECT_101)
        return padded

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get data
        img = self.images[idx]  # 101x101 uint8
        depth = self.depths[idx]  # float
        img_id = self.ids[idx]

        if self.masks is not None:
            mask = self.masks[idx]  # 101x101 uint8
        else:
            mask = np.zeros_like(img)

        # Augmentation (Horizontal Flip) - Only for train
        if self.mode == "train" and np.random.rand() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        # Padding (101x101 -> 128x128)
        # copyMakeBorder handles non-contiguous arrays from fliplr automatically
        img_padded = self._reflection_pad(img, target_size=128)
        mask_padded = self._reflection_pad(mask, target_size=128)

        # Normalize and Convert to Tensor
        # Image: (H, W) -> (1, H, W), float 0-1
        img_tensor = torch.from_numpy(img_padded).float().unsqueeze(0) / 255.0

        # Mask: (H, W) -> (1, H, W), float 0-1
        mask_tensor = torch.from_numpy(mask_padded).float().unsqueeze(0)
        mask_tensor = (mask_tensor > 0.5).float()

        # Depth: float tensor
        depth_tensor = torch.tensor([depth], dtype=torch.float32)

        return img_tensor, mask_tensor, depth_tensor, img_id
