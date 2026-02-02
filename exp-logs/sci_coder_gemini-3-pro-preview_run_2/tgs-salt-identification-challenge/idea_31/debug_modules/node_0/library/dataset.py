import os
import hashlib
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode, pad_image


def get_transforms(phase: str):
    """
    Returns the Albumentations transformations for the given phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    # Calculate 1-channel mean/std from Config (which is RGB)
    # We average the RGB values to get a reasonable grayscale approximation
    mean_val = float(np.mean(Config.MEAN))
    std_val = float(np.mean(Config.STD))

    transforms = []

    if phase == "train":
        # Non-Rigid: Elastic Transform
        transforms.append(
            A.ElasticTransform(
                alpha=Config.AUG_ELASTIC_ALPHA,
                sigma=Config.AUG_ELASTIC_SIGMA,
                alpha_affine=Config.AUG_ELASTIC_ALPHA,
                p=Config.AUG_ELASTIC_P,
            )
        )

        # Rigid: ShiftScaleRotate
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=Config.AUG_RIGID_P,
            )
        )

        # Horizontal Flip
        transforms.append(A.HorizontalFlip(p=0.5))

    # Normalization and Tensor conversion (Common to all phases)
    # Note: We assume input is 1-channel.
    transforms.append(
        A.Normalize(mean=(mean_val,), std=(std_val,), max_pixel_value=255.0)
    )
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


class SaltDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        mode: str = "train",
        transform: A.Compose = None,
        depth_stats: tuple = None,
        load_cached: bool = True,
        cache_name: str = None,
    ):
        """
        Dataset for Salt Segmentation.

        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            depth_stats (tuple): (mean, std) for depth normalization. If None and mode='train', calculated from df.
            load_cached (bool): Whether to try loading from cache.
            cache_name (str): Optional unique identifier for cache (e.g. 'fold0'). If None, uses mode.
        """
        self.df = df
        self.mode = mode
        self.transform = transform

        # Determine Cache Path
        if cache_name is None:
            cache_name = mode

        # Create a hash of the IDs to verify or append to cache name to avoid collisions
        # This ensures that if we use a subset (debug) or different fold, we get a unique cache
        ids_hash = hashlib.md5(pd.util.hash_pandas_object(df["id"]).values).hexdigest()[
            :8
        ]
        self.cache_dir = os.path.join(
            Config.WORKING_DIR, f"cache_{cache_name}_{ids_hash}"
        )

        self.images = None
        self.masks = None
        self.depths = None
        self.ids = df["id"].values

        # 1. Data Loading / Caching
        if load_cached and self._check_cache_exists():
            # print(f"Loading cached data from {self.cache_dir}...")
            self._load_cache()
        else:
            # print(f"Processing data and saving to {self.cache_dir}...")
            self._process_and_cache()

        # 2. Depth Statistics
        if depth_stats is not None:
            self.depth_mean, self.depth_std = depth_stats
        elif mode == "train":
            self.depth_mean = float(np.mean(self.depths))
            self.depth_std = float(np.std(self.depths))
            print(
                f"Calculated Depth Stats: Mean={self.depth_mean:.10f}, Std={self.depth_std:.10f}"
            )
        else:
            # Fallback if not provided for val/test (should ideally be passed from train)
            self.depth_mean = float(np.mean(self.depths))
            self.depth_std = float(np.std(self.depths))

    def _check_cache_exists(self):
        """Checks if all required npy files exist in the cache directory."""
        required = ["images.npy", "depths.npy"]
        if self.mode in ["train", "val"]:
            required.append("masks.npy")

        for f in required:
            if not os.path.exists(os.path.join(self.cache_dir, f)):
                return False
        return True

    def _process_and_cache(self):
        """Loads images/masks from disk, processes them, and saves to npy cache."""
        os.makedirs(self.cache_dir, exist_ok=True)

        img_list = []
        mask_list = []
        depth_list = []

        for idx, row in self.df.iterrows():
            # Load Image
            img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            # Pad Image (101x101 -> 128x128)
            img = pad_image(img)
            img_list.append(img)

            # Load Mask (if applicable)
            if self.mode in ["train", "val"]:
                rle = row["rle_mask"]
                # Decode RLE to original size then pad
                mask = rle_decode(rle, shape=(Config.ORIG_SIZE, Config.ORIG_SIZE))
                mask = pad_image(mask)
                mask_list.append(mask)

            # Load Depth
            depth_list.append(row["z"])

        # Convert to numpy arrays
        # Images: (N, H, W)
        self.images = np.array(img_list, dtype=np.uint8)
        self.depths = np.array(depth_list, dtype=np.float32)

        np.save(os.path.join(self.cache_dir, "images.npy"), self.images)
        np.save(os.path.join(self.cache_dir, "depths.npy"), self.depths)

        if self.mode in ["train", "val"]:
            self.masks = np.array(mask_list, dtype=np.uint8)
            np.save(os.path.join(self.cache_dir, "masks.npy"), self.masks)

    def _load_cache(self):
        """Loads data from npy cache."""
        self.images = np.load(os.path.join(self.cache_dir, "images.npy"))
        self.depths = np.load(os.path.join(self.cache_dir, "depths.npy"))
        if self.mode in ["train", "val"]:
            self.masks = np.load(os.path.join(self.cache_dir, "masks.npy"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Retrieve data from memory
        image = self.images[idx]  # (H, W)
        depth = self.depths[idx]
        img_id = self.ids[idx]

        mask = None
        if self.mode in ["train", "val"]:
            mask = self.masks[idx]  # (H, W)

        # Prepare for Albumentations
        # Expand image to (H, W, 1) for consistent channel handling
        image = np.expand_dims(image, axis=-1)

        # Apply Augmentations
        if self.transform:
            if mask is not None:
                # Albumentations handles (H, W) masks correctly
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # Depth Normalization: (z - mean) / std
        if self.depth_std > 1e-6:
            depth = (depth - self.depth_mean) / self.depth_std
        else:
            depth = depth - self.depth_mean

        # Convert depth to tensor
        depth_tensor = torch.tensor([depth], dtype=torch.float32)

        if self.mode in ["train", "val"]:
            # Ensure mask is a float tensor with channel dim (1, H, W)
            # ToTensorV2 usually returns (H, W) for masks if input was (H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()
            return image, mask, depth_tensor, img_id
        else:
            return image, depth_tensor, img_id
