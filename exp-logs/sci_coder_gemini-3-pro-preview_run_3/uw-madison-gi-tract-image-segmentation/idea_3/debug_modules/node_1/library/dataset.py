import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode


def process_metadata(csv_path, mode="train", load_cached_data=True):
    """
    Loads metadata and prepares it for 2.5D training (calculating prev/next slice paths).
    Implements caching to Parquet to speed up subsequent runs.
    """
    cache_filename = f"{mode}_df_25d.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    df = pd.read_csv(csv_path)

    # Ensure slice is integer for proper sorting
    df["slice"] = df["slice"].astype(int)

    # Sort to ensure correct ordering for temporal shifting
    df = df.sort_values(["case", "day", "slice"]).reset_index(drop=True)

    # Group by case and day to prevent shifting across different scans
    # We use transform to keep the index aligned
    # Shift +1 for previous, -1 for next (since we sorted ascending)
    # Note: shift(1) gets the row with index i-1, which is the previous slice
    grouped = df.groupby(["case", "day"])

    df["image_path_prev"] = grouped["image_path"].shift(1)
    df["image_path_next"] = grouped["image_path"].shift(-1)

    # Handle boundary conditions (first and last slices)
    # If prev is NaN, use current. If next is NaN, use current.
    df["image_path_prev"] = df["image_path_prev"].fillna(df["image_path"])
    df["image_path_next"] = df["image_path_next"].fillna(df["image_path"])

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_transforms(data):
    """
    Returns albumentations transforms based on the data mode.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(
                    Config.IMG_SIZE[0],
                    Config.IMG_SIZE[1],
                    interpolation=cv2.INTER_LINEAR,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    ],
                    p=0.25,
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(
                    Config.IMG_SIZE[0],
                    Config.IMG_SIZE[1],
                    interpolation=cv2.INTER_LINEAR,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(
                    Config.IMG_SIZE[0],
                    Config.IMG_SIZE[1],
                    interpolation=cv2.INTER_LINEAR,
                ),
                ToTensorV2(),
            ]
        )


class UWMadissonDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata and paths.
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'valid', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.input_dir = Config.INPUT_DIR
        self.classes = Config.CLASSES

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        """
        Loads an image from path.
        """
        full_path = os.path.join(self.input_dir, path)
        # Load as-is (likely 16-bit or 8-bit grayscale)
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # Fallback for safety, though paths should be valid
            return np.zeros((Config.IMG_SIZE[0], Config.IMG_SIZE[1]), dtype=np.float32)

        # Ensure image is 2D (H, W)
        if len(img.shape) == 3:
            img = img[:, :, 0]

        return img.astype(np.float32)

    def _normalize_slice(self, img):
        """
        Applies Robust Per-Slice Normalization.
        Clips to [p1, p99] and scales to [0, 1].
        """
        # Calculate percentiles
        p_min = np.percentile(img, Config.NORM_MIN_PERCENTILE)
        p_max = np.percentile(img, Config.NORM_MAX_PERCENTILE)

        # Clip
        img_clipped = np.clip(img, p_min, p_max)

        # Min-Max Scale
        if p_max - p_min > 0:
            img_norm = (img_clipped - p_min) / (p_max - p_min)
        else:
            img_norm = np.zeros_like(img_clipped)

        return img_norm

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load 2.5D Images (Prev, Curr, Next)
        path_prev = row["image_path_prev"]
        path_curr = row["image_path"]
        path_next = row["image_path_next"]

        img_prev = self._load_image(path_prev)
        img_curr = self._load_image(path_curr)
        img_next = self._load_image(path_next)

        # 2. Normalize each slice independently
        img_prev = self._normalize_slice(img_prev)
        img_curr = self._normalize_slice(img_curr)
        img_next = self._normalize_slice(img_next)

        # 3. Stack to create (H, W, 3) image
        # Channels last for albumentations
        image = np.stack([img_prev, img_curr, img_next], axis=-1)

        # 4. Handle Masks (Train/Valid only)
        if self.mode in ["train", "valid"]:
            # Original image dimensions
            h, w = row["height"], row["width"]

            masks = []
            for cls in self.classes:
                rle = row[cls]
                mask = rle_decode(rle, shape=(h, w))
                masks.append(mask)

            # Stack masks -> (H, W, C)
            mask = np.stack(masks, axis=-1)

            # Apply augmentations
            if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            # Mask is already (C, H, W) after ToTensorV2 if it was (H, W, C) input?
            # ToTensorV2 converts image to (C, H, W) and mask to (H, W, C) usually unless transposed.
            # Actually ToTensorV2 preserves mask shape (H, W, C) but converts to tensor.
            # We usually want (C, H, W) for PyTorch models.
            # Albumentations ToTensorV2 behavior:
            # If mask is passed, it returns a tensor. If mask is (H, W, C), it returns (H, W, C).
            # We need to manually permute mask to (C, H, W).
            mask = mask.permute(2, 0, 1).float()

            return {
                "image": image.float(),  # (3, H, W)
                "mask": mask,  # (3, H, W)
                "id": row["id"],
            }

        else:
            # Inference Mode
            if self.transforms:
                augmented = self.transforms(image=image)
                image = augmented["image"]

            return {
                "image": image.float(),
                "id": row["id"],
                "case": row["case"],
                "day": row["day"],
                "slice": row["slice"],
                "height": row["height"],
                "width": row["width"],
            }
