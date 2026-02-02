import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode, pad_image

# Constants for Normalization
# Using Red channel stats of ImageNet as a proxy for grayscale intensity
MEAN = [0.485]
STD = [0.229]


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline for the given mode.
    """
    if mode == "train":
        return A.Compose(
            [
                # Non-Rigid Augmentation (Critical)
                A.ElasticTransform(
                    alpha=Config.AUG_ELASTIC_ALPHA,
                    sigma=Config.AUG_ELASTIC_SIGMA,
                    alpha_affine=Config.AUG_ELASTIC_ALPHA_AFFINE,
                    p=Config.AUG_ELASTIC_P,  # Reduced probability (Cite {solution_lesson_node_00073})
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                # Rigid Augmentation
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=Config.AUG_RIGID_P,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                A.HorizontalFlip(p=0.5),
                # Normalization & Tensor conversion
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )
    else:
        # Val / Test / Pseudo
        return A.Compose(
            [
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )


class SaltDataset(Dataset):
    def __init__(
        self, images, masks=None, depths=None, ids=None, transform=None, mode="train"
    ):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W).
            masks (np.ndarray, optional): Array of masks (N, H, W). Binary or Soft.
            depths (np.ndarray, optional): Array of depths (N,).
            ids (np.ndarray or list, optional): List of image IDs.
            transform (A.Compose): Albumentations transform.
            mode (str): 'train', 'val', 'test', 'pseudo'.
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Ensure image is (H, W, C) for Albumentations
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        data = {"image": image}

        if self.masks is not None:
            mask = self.masks[idx]
            data["mask"] = mask

        if self.transform:
            augmented = self.transform(**data)
            image = augmented["image"]
            if "mask" in augmented:
                mask = augmented["mask"]
                # Convert to float tensor (BCE requires float, even for binary targets)
                # Shape: (1, H, W)
                mask = mask.float().unsqueeze(0)

        # Handle Depth
        depth = 0.0
        if self.depths is not None:
            depth = torch.tensor([self.depths[idx]], dtype=torch.float32)

        # Get ID
        img_id = self.ids[idx] if self.ids is not None else ""

        if self.mode == "test":
            return image, depth, img_id

        # For train/val/pseudo
        # If masks are missing (shouldn't happen in valid flow), return dummy
        if "mask" not in locals():
            mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)

        return image, mask, depth, img_id


def process_dataframe(df, mode, config):
    """
    Reads images and masks from disk based on the dataframe paths.
    Pads them to the target size.
    """
    images = []
    masks = []
    depths = []

    # Using df.itertuples for slightly better performance than iterrows
    for row in df.itertuples():
        img_path = os.path.join(config.INPUT_DIR, row.image_path)

        # Read Image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Should not happen given metadata verification
            continue

        # Pad Image
        img = pad_image(img, target_size=config.IMG_SIZE)
        images.append(img)

        # Read Mask (if exists)
        # Check if rle_mask exists and is not NaN
        if hasattr(row, "rle_mask") and not pd.isna(row.rle_mask):
            mask = rle_decode(row.rle_mask, shape=(config.ORIG_SIZE, config.ORIG_SIZE))
            mask = pad_image(mask, target_size=config.IMG_SIZE)
            masks.append(mask)
        elif mode in ["train", "val"]:
            # If train/val but no mask, assume empty (all zeros)
            mask = np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)
            masks.append(mask)

        # Depth
        if hasattr(row, "z"):
            depths.append(row.z)
        else:
            depths.append(0)

    return np.array(images), np.array(masks), np.array(depths)


def load_data(mode="train", load_cached_data=True):
    """
    Loads data for the specified mode.
    Handles caching of processed numpy arrays to disk.
    Applies depth normalization based on training set statistics.
    """
    # Determine CSV path
    if mode == "train":
        csv_path = Config.TRAIN_CSV
    elif mode == "val":
        csv_path = Config.VAL_CSV
    elif mode == "test":
        csv_path = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Read Metadata
    df = pd.read_csv(csv_path)
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    # Extract IDs directly from dataframe (metadata is source of truth)
    ids = df["id"].values

    # Define Cache Paths
    cache_base = os.path.join(Config.CACHE_DIR, f"{mode}_data")
    img_cache = cache_base + "_images.npy"
    mask_cache = cache_base + "_masks.npy"
    depth_cache = cache_base + "_depths.npy"
    depth_stats_path = os.path.join(Config.CACHE_DIR, "depth_stats.npy")

    # Initialize variables
    images = None
    masks = None
    depths = None

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(img_cache) and os.path.exists(depth_cache):
        try:
            images = np.load(img_cache)
            depths = np.load(depth_cache)
            if mode in ["train", "val"]:
                if os.path.exists(mask_cache):
                    masks = np.load(mask_cache)
                else:
                    # Partial cache, force reload
                    images = None
        except Exception:
            # Corrupt cache, force reload
            images = None

    # 2. Process from Scratch if needed
    if images is None:
        images, masks, depths = process_dataframe(df, mode, Config)

        # Save to Cache
        np.save(img_cache, images)
        np.save(depth_cache, depths)
        if masks is not None and len(masks) > 0:
            np.save(mask_cache, masks)

    # 3. Handle Depth Normalization
    # We standardize depth based on Training set statistics
    if mode == "train":
        d_mean = np.mean(depths)
        d_std = np.std(depths)
        np.save(depth_stats_path, np.array([d_mean, d_std]))
    else:
        if os.path.exists(depth_stats_path):
            stats = np.load(depth_stats_path)
            d_mean, d_std = stats[0], stats[1]
        else:
            # Fallback (should not happen if train runs first)
            d_mean = np.mean(depths)
            d_std = np.std(depths)

    # Apply Standard Scaling
    depths = (depths - d_mean) / (d_std + 1e-8)

    # Return Dataset
    return SaltDataset(
        images=images,
        masks=masks if mode in ["train", "val"] else None,
        depths=depths,
        ids=ids,
        transform=get_transforms(mode),
        mode=mode,
    )
