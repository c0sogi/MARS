import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    CACHE_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    AUG_ELASTIC_ALPHA,
    AUG_ELASTIC_SIGMA,
    AUG_ELASTIC_PROB,
    AUG_RIGID_PROB,
    SEED,
)
from library.utils import pad_image, rle_decode

# Standard ImageNet statistics (using Red channel for 1-channel input)
IMAGENET_MEAN = [0.485]
IMAGENET_STD = [0.229]


def get_depth_stats(load_cached_data=True):
    """
    Calculates or loads global depth statistics (mean and std) for standardization.
    Caches the result to ensure consistency between training and inference.
    """
    cache_path = os.path.join(CACHE_DIR, "depth_stats.npy")

    if load_cached_data and os.path.exists(cache_path):
        stats = np.load(cache_path, allow_pickle=True).item()
        return stats["mean"], stats["std"]

    # Load all available depth data to compute global stats
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    depths = []
    for p in [train_path, val_path, test_path]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "z" in df.columns:
                depths.extend(df["z"].tolist())

    depths = np.array(depths)
    mean_val = np.nanmean(depths)
    std_val = np.nanstd(depths)

    # Save to cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path, {"mean": mean_val, "std": std_val})

    return mean_val, std_val


def get_transforms(phase="train"):
    """
    Returns the Albumentations transformation pipeline for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                # Non-rigid deformations (Crucial for salt)
                A.ElasticTransform(
                    alpha=AUG_ELASTIC_ALPHA,
                    sigma=AUG_ELASTIC_SIGMA,
                    alpha_affine=None,
                    p=AUG_ELASTIC_PROB,
                ),
                # Rigid geometric augmentations
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=AUG_RIGID_PROB,
                ),
                # Flip
                A.HorizontalFlip(p=0.5),
                # Normalization and Tensor conversion
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / Inference
        return A.Compose(
            [
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles loading images, masks, and depth information.
    Supports binary masks (train) and soft pseudo-labels (distillation).
    """

    def __init__(
        self,
        mode="train",
        pseudo_labels=None,
        transform=None,
        debug_size=None,
    ):
        """
        Args:
            mode (str): 'train', 'val', 'test', or 'semi_supervised'.
            pseudo_labels (dict, optional): Dict mapping ID to soft mask (numpy array) for distillation.
            transform (albumentations.Compose): Transformations to apply.
            debug_size (int, optional): Limit dataset size for debugging.
        """
        self.mode = mode
        self.pseudo_labels = pseudo_labels
        self.transform = transform

        # Load Metadata based on mode
        self.df = self._load_metadata(mode)

        if debug_size is not None:
            self.df = self.df.iloc[:debug_size]

        # Load Depth Stats for normalization
        self.depth_mean, self.depth_std = get_depth_stats(load_cached_data=True)

    def _load_metadata(self, mode):
        if mode == "train":
            return pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
        elif mode == "val":
            return pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
        elif mode == "test":
            return pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))
        elif mode == "semi_supervised":
            # Combine Train and Test for Stage 3
            train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
            val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
            test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))
            # Concatenate all
            return pd.concat([train_df, val_df, test_df], ignore_index=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["id"]

        # 1. Load Image
        # Images are grayscale. Load as such.
        image_path = os.path.join(INPUT_DIR, row["image_path"])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 2. Pad Image (101x101 -> 128x128)
        # We pad BEFORE augmentation to give context for elastic transforms
        image = pad_image(image)

        # 3. Load Mask (if available)
        mask = None

        # Priority: Pseudo-labels (for distillation on test set)
        if self.pseudo_labels is not None and image_id in self.pseudo_labels:
            mask = self.pseudo_labels[image_id]
            # Ensure mask is padded if it came from raw prediction,
            # but usually pseudo-labels are generated at 128x128.
            # We assume pseudo-labels match the IMG_HEIGHT/IMG_WIDTH.
            if mask.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
                # If pseudo-label is 101x101, pad it
                mask = pad_image(mask)

            # Ensure float32 for soft targets
            mask = mask.astype(np.float32)

        # Fallback: Ground Truth from RLE (for train/val)
        elif "rle_mask" in row and pd.notna(row["rle_mask"]):
            mask = rle_decode(row["rle_mask"])
            mask = pad_image(mask)
            mask = mask.astype(np.float32)

        # Fallback: Empty mask (for test set without pseudo-labels)
        else:
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

        # 4. Apply Augmentations
        if self.transform:
            # Albumentations expects H,W,C or H,W
            # Our image is H,W. Mask is H,W.
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 5. Process Depth
        # Standard Scaling: (z - mean) / std
        z = row["z"]
        z_norm = (z - self.depth_mean) / self.depth_std
        if np.isnan(z_norm):
            z_norm = 0.0
        z_tensor = torch.tensor([z_norm], dtype=torch.float32)

        # 6. Final Formatting
        # Image is already Tensor (C, H, W) from ToTensorV2
        # Mask needs to be Tensor (H, W) or (1, H, W)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)

        # Add channel dim to mask if missing: (H, W) -> (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {"image": image, "mask": mask, "depth": z_tensor, "id": image_id}
