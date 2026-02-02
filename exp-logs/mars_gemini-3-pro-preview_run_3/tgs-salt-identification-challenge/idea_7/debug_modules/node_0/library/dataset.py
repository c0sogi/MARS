import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline based on the mode.
    Implements Conservative Geometric Regularization and Input Normalization.
    """
    transforms = []

    # 1. Padding to IMG_SIZE (128x128) using Reflection
    transforms.append(
        A.PadIfNeeded(
            min_height=Config.IMG_SIZE,
            min_width=Config.IMG_SIZE,
            border_mode=cv2.BORDER_REFLECT_101,
            value=0,
            mask_value=0,
            p=1.0,
        )
    )

    # 2. Augmentations (Train only)
    if mode == "train":
        transforms.append(A.HorizontalFlip(p=0.5))

        # Conservative Geometric Regularization
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=Config.AUG_SHIFT,
                scale_limit=Config.AUG_SCALE,
                rotate_limit=Config.AUG_ROTATION,
                border_mode=cv2.BORDER_REFLECT_101,
                value=0,
                mask_value=0,
                p=Config.AUG_PROB,
            )
        )

        transforms.append(
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2)
        )

    # 3. Normalization (ImageNet stats for 3-channel input)
    # Input is expected to be [Seismic, Seismic, Depth] in range [0, 1]
    transforms.append(
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=1.0,  # We will pre-normalize inputs to 0-1 float
            p=1.0,
        )
    )

    # 4. Convert to Tensor
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


class SaltDataset(Dataset):
    """
    Dataset class for Salt Segmentation.
    Handles caching, channel multiplexing, and loading.
    """

    def __init__(self, mode="train", transform=None, load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transform pipeline.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        self.mode = mode
        self.transform = transform

        # Determine metadata path
        if mode == "train":
            self.metadata_path = Config.TRAIN_METADATA
        elif mode == "val":
            self.metadata_path = Config.VAL_METADATA
        elif mode == "test":
            self.metadata_path = Config.TEST_METADATA
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Load global depth stats for consistent normalization
        self._load_global_depth_stats()

        # Define cache paths
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_paths = {
            "images": os.path.join(self.cache_dir, f"cached_{mode}_images.npy"),
            "masks": os.path.join(self.cache_dir, f"cached_{mode}_masks.npy"),
            "depths": os.path.join(self.cache_dir, f"cached_{mode}_depths.npy"),
            "ids": os.path.join(self.cache_dir, f"cached_{mode}_ids.npy"),
        }

        # Load Data (Cache or Compute)
        self.images, self.masks, self.depths, self.ids = self._load_data(
            load_cached_data
        )

    def _load_global_depth_stats(self):
        """Calculates global min/max depth from depths.csv."""
        try:
            df = pd.read_csv(Config.DEPTHS_CSV)
            self.min_depth = df["z"].min()
            self.max_depth = df["z"].max()
        except Exception as e:
            print(
                f"Warning: Could not load depths.csv for global stats. Using defaults. Error: {e}"
            )
            self.min_depth = 0
            self.max_depth = 1000

    def _load_data(self, load_cached_data):
        """
        Loads data from cache if available and requested.
        Otherwise, loads from disk, processes, and saves to cache.
        """
        # Check if all cache files exist
        cache_exists = all(os.path.exists(p) for p in self.cache_paths.values())

        if load_cached_data and cache_exists:
            print(f"[{self.mode.upper()}] Loading data from cache...")
            images = np.load(self.cache_paths["images"])
            masks = np.load(self.cache_paths["masks"])
            depths = np.load(self.cache_paths["depths"])
            ids = np.load(self.cache_paths["ids"])
            return images, masks, depths, ids

        # If not cached or reload forced, process from scratch
        print(
            f"[{self.mode.upper()}] Cache miss or reload forced. Processing raw data..."
        )

        df = pd.read_csv(self.metadata_path)

        images_list = []
        masks_list = []
        depths_list = []
        ids_list = []

        for idx, row in df.iterrows():
            # 1. Load Image
            img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            # 2. Load Mask (if available)
            mask = None
            if self.mode in ["train", "val"]:
                mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                # Binarize mask (0 or 255 -> 0 or 1)
                mask = (mask > 127).astype(np.uint8)
            else:
                # Create dummy mask for test
                mask = np.zeros_like(img, dtype=np.uint8)

            # 3. Store Depth
            z = row["z"]

            images_list.append(img)
            masks_list.append(mask)
            depths_list.append(z)
            ids_list.append(str(row["id"]))

        # Convert to numpy arrays
        images = np.array(images_list, dtype=np.uint8)
        masks = np.array(masks_list, dtype=np.uint8)
        depths = np.array(depths_list, dtype=np.float32)
        ids = np.array(ids_list)

        # Save to cache
        print(f"[{self.mode.upper()}] Saving processed data to cache...")
        np.save(self.cache_paths["images"], images)
        np.save(self.cache_paths["masks"], masks)
        np.save(self.cache_paths["depths"], depths)
        np.save(self.cache_paths["ids"], ids)

        return images, masks, depths, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 1. Retrieve raw data
        img_raw = self.images[idx]  # (101, 101) uint8
        mask_raw = self.masks[idx]  # (101, 101) uint8
        depth_raw = self.depths[idx]  # scalar
        img_id = self.ids[idx]

        # 2. Normalize raw image to [0, 1]
        img_norm = img_raw.astype(np.float32) / 255.0

        # 3. Normalize depth to [0, 1]
        depth_norm = (depth_raw - self.min_depth) / (
            self.max_depth - self.min_depth + 1e-6
        )

        # 4. Construct 3-Channel Input [Seismic, Seismic, Depth]
        # Create depth channel of same shape as image
        depth_channel = np.full_like(img_norm, depth_norm, dtype=np.float32)

        # Stack channels (H, W, 3)
        combined_img = np.dstack([img_norm, img_norm, depth_channel])

        # 5. Apply Augmentations
        if self.transform:
            augmented = self.transform(image=combined_img, mask=mask_raw)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"]
        else:
            # Fallback if no transform (should not happen in this pipeline)
            transforms = ToTensorV2()
            augmented = transforms(image=combined_img, mask=mask_raw)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"]

        # 6. Post-process Mask
        # ToTensorV2 converts (H, W) numpy to (H, W) tensor if no channel dim.
        # We need (1, H, W) float tensor for BCE/Dice loss.
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        mask_tensor = mask_tensor.float()

        return image_tensor, mask_tensor, img_id
