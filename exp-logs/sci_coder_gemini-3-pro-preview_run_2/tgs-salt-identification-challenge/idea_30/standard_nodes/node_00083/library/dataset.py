import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import rle_decode

# Constants
INPUT_ROOT = "./input"
CACHE_DIR = "./working/idea_30"
ORIG_SIZE = 101
TARGET_SIZE = 128

# ImageNet stats averaged for 1-channel grayscale
# Mean: (0.485 + 0.456 + 0.406) / 3 = 0.449
# Std: (0.229 + 0.224 + 0.225) / 3 = 0.226
IMAGENET_MEAN_1CH = [0.449]
IMAGENET_STD_1CH = [0.226]


def get_depth_stats(load_cached_data=True):
    """
    Calculates or loads depth statistics (mean, std) from the training set.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "depth_stats.csv")

    if load_cached_data and os.path.exists(cache_path):
        stats_df = pd.read_csv(cache_path)
        return stats_df.iloc[0]["mean"], stats_df.iloc[0]["std"]

    # Load training metadata to compute stats
    train_csv_path = "./metadata/train.csv"
    if not os.path.exists(train_csv_path):
        # Fallback if metadata not generated yet (should not happen based on prompt)
        raise FileNotFoundError(f"Metadata file {train_csv_path} not found.")

    df = pd.read_csv(train_csv_path)
    depths = df["z"].values
    d_mean = np.mean(depths)
    d_std = np.std(depths)

    # Save to cache
    stats_df = pd.DataFrame({"mean": [d_mean], "std": [d_std]})
    stats_df.to_csv(cache_path, index=False)

    return d_mean, d_std


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specified phase.
    """
    # Calculate padding to reach 128x128 from 101x101
    pad_h = TARGET_SIZE - ORIG_SIZE
    pad_w = TARGET_SIZE - ORIG_SIZE
    # Pad equally on sides if possible, but 27 is odd.
    # PadTop=13, PadBottom=14, PadLeft=13, PadRight=14
    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l

    if phase == "train":
        return A.Compose(
            [
                # Spatial Alignment: Pad to 128x128 using reflection
                A.PadIfNeeded(
                    min_height=TARGET_SIZE,
                    min_width=TARGET_SIZE,
                    border_mode=cv2.BORDER_REFLECT_101,
                    value=0,
                    mask_value=0,
                    always_apply=True,
                ),
                # Non-Rigid Augmentation: Elastic Transform
                # alpha ~ 120, sigma ~ 6
                A.ElasticTransform(
                    alpha=120, sigma=6, alpha_affine=3.6, p=0.5  # approx 0.03 * alpha
                ),
                # Rigid Augmentation
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
                ),
                A.HorizontalFlip(p=0.5),
                # Normalization
                A.Normalize(mean=IMAGENET_MEAN_1CH, std=IMAGENET_STD_1CH),
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=TARGET_SIZE,
                    min_width=TARGET_SIZE,
                    border_mode=cv2.BORDER_REFLECT_101,
                    value=0,
                    mask_value=0,
                    always_apply=True,
                ),
                A.Normalize(mean=IMAGENET_MEAN_1CH, std=IMAGENET_STD_1CH),
                ToTensorV2(transpose_mask=True),
            ]
        )


class SaltDataset(Dataset):
    def __init__(self, mode="train", fold=None, debug=False):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            fold (int): Optional, for k-fold (not used here as splits are pre-defined in metadata).
            debug (bool): If True, limits dataset size for debugging.
        """
        self.mode = mode
        self.debug = debug

        # Load Metadata
        if mode == "train":
            self.df = pd.read_csv("./metadata/train.csv")
        elif mode == "val":
            self.df = pd.read_csv("./metadata/val.csv")
        elif mode == "test":
            self.df = pd.read_csv("./metadata/test.csv")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if self.debug:
            self.df = self.df.head(50)

        # Load Depth Stats
        self.depth_mean, self.depth_std = get_depth_stats(load_cached_data=True)

        # Transforms
        self.transforms = get_transforms(phase="train" if mode == "train" else "val")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["id"]

        # Load Image
        # Metadata contains relative path e.g., "train/images/xxxx.png"
        img_path = os.path.join(INPUT_ROOT, row["image_path"])

        # Load as grayscale (1 channel)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Ensure image is (H, W, 1) for Albumentations
        image = np.expand_dims(image, axis=-1)

        # Load Mask (if exists)
        mask = None
        if self.mode in ["train", "val"]:
            rle = row["rle_mask"]
            # Decode RLE
            mask = rle_decode(rle, shape=(ORIG_SIZE, ORIG_SIZE))
            # Expand dims for albumentations
            mask = np.expand_dims(mask, axis=-1)

        # Normalize Depth
        # z is in the dataframe
        z_raw = row["z"]
        z_norm = (z_raw - self.depth_mean) / self.depth_std
        # Convert to float32 tensor
        z_tensor = torch.tensor([z_norm], dtype=torch.float32)

        # Apply Augmentations
        if mask is not None:
            augmented = self.transforms(image=image, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"]

            # Mask comes out as [1, H, W] or [H, W] depending on ToTensorV2 config
            # We want [1, H, W] float for BCE/Lovasz
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)

            # Binarize mask to 0.0 / 1.0
            mask_tensor = mask_tensor.float()

            return image_tensor, mask_tensor, z_tensor, image_id
        else:
            # Test mode
            augmented = self.transforms(image=image)
            image_tensor = augmented["image"]

            return image_tensor, z_tensor, image_id
