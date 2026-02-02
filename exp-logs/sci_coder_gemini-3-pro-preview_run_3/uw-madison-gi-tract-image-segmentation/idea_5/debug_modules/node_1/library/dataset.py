import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG
from library.utils import rle_decode


def robust_normalize(img):
    """
    Apply robust normalization to a single slice.
    Clips to 1st and 99th percentiles, then min-max scales to [0, 1].
    """
    img = img.astype(np.float32)
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)
    img = np.clip(img, p1, p99)

    denom = p99 - p1
    if denom == 0:
        denom = 1e-6

    img = (img - p1) / denom
    return img


def resize_pad_square(img, mask=None, target_size=None):
    """
    Resizes the longest dimension to target_size and pads the shorter dimension
    to preserve aspect ratio.

    Args:
        img: (H, W, C) or (H, W) numpy array
        mask: (H, W, C) numpy array or None
        target_size: tuple (H, W), defaults to CFG.img_size
    """
    if target_size is None:
        target_size = CFG.img_size

    target_h, target_w = target_size
    h, w = img.shape[:2]

    # Calculate scaling factor to fit the longest dimension
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if len(img_resized.shape) == 2:
        img_resized = np.expand_dims(img_resized, axis=-1)

    if mask is not None:
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if len(mask_resized.shape) == 2:
            mask_resized = np.expand_dims(mask_resized, axis=-1)
    else:
        mask_resized = None

    # Pad to reach target size (pad right and bottom)
    delta_h = target_h - new_h
    delta_w = target_w - new_w

    # Center padding
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )

    if mask is not None:
        mask_padded = cv2.copyMakeBorder(
            mask_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
        return img_padded, mask_padded

    return img_padded


def process_25d_dataframe(df, split_name="train", load_cached_data=True):
    """
    Process dataframe to add 2.5D context paths (prev, next slices).
    Implements caching mechanism.
    """
    cache_file = f"{split_name}_metadata_25d.parquet"
    cache_path = os.path.join(CFG.working_dir, cache_file)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached 2.5D metadata from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing {split_name} metadata for 2.5D context...")

    # Ensure slice is int for sorting
    df = df.copy()
    df["slice_int"] = df["slice"].astype(int)

    # Sort by case, day, slice
    df = df.sort_values(["case", "day", "slice_int"]).reset_index(drop=True)

    # Group by scan (case + day)
    groups = df.groupby(["case", "day"])

    # Shift to get prev and next paths
    df["image_path_prev"] = groups["image_path"].shift(1)
    df["image_path_next"] = groups["image_path"].shift(-1)

    # Fill missing neighbors (boundaries) with current slice path
    df["image_path_prev"] = df["image_path_prev"].fillna(df["image_path"])
    df["image_path_next"] = df["image_path_next"].fillna(df["image_path"])

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path)

    return df


class UWMGIDataset(Dataset):
    def __init__(self, df, label=True, transforms=None):
        """
        Args:
            df: DataFrame containing metadata and 2.5D paths.
            label: Boolean, whether to return masks.
            transforms: Albumentations transforms.
        """
        self.df = df
        self.label = label
        self.transforms = transforms
        self.base_path = CFG.base_input_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # 1. Load 2.5D Images (t-1, t, t+1)
        paths = [row["image_path_prev"], row["image_path"], row["image_path_next"]]
        images = []

        for p in paths:
            full_path = os.path.join(self.base_path, p)
            # Load as-is (likely 16-bit or 8-bit grayscale)
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

            # Handle case where image load fails
            if img is None:
                # Fallback to zeros if file missing (should not happen with valid metadata)
                img = np.zeros((CFG.img_size[0], CFG.img_size[1]), dtype=np.uint8)

            # Ensure 2D
            if len(img.shape) > 2:
                img = img[..., 0]  # Take first channel if multi-channel read

            # Normalize per slice
            img = robust_normalize(img)
            images.append(img)

        # Stack to (H, W, 3)
        img_stack = np.stack(images, axis=-1)  # (H, W, 3)

        # 2. Load Masks (if label=True)
        mask_stack = None
        if self.label:
            h, w = img_stack.shape[:2]
            masks = []
            classes = ["large_bowel", "small_bowel", "stomach"]

            for cls in classes:
                rle = row[cls] if cls in row else ""
                mask = rle_decode(rle, shape=(h, w))
                masks.append(mask)

            mask_stack = np.stack(masks, axis=-1)  # (H, W, 3)

        # 3. Resize and Pad (Geometry Preserving)
        if self.label:
            img_stack, mask_stack = resize_pad_square(
                img_stack, mask_stack, tuple(CFG.img_size)
            )
        else:
            img_stack = resize_pad_square(img_stack, None, tuple(CFG.img_size))

        # 4. Apply Transforms (Augmentations)
        if self.transforms:
            if self.label:
                data = self.transforms(image=img_stack, mask=mask_stack)
                img_stack = data["image"]
                mask_stack = data["mask"]
            else:
                data = self.transforms(image=img_stack)
                img_stack = data["image"]

        # 5. Return
        # Ensure float32 for model
        if not isinstance(img_stack, torch.Tensor):
            img_stack = torch.from_numpy(img_stack.transpose(2, 0, 1)).float()

        if self.label:
            if isinstance(mask_stack, torch.Tensor):
                mask_stack = mask_stack.permute(2, 0, 1).float()
            else:
                mask_stack = torch.from_numpy(mask_stack.transpose(2, 0, 1)).float()
            return img_stack, mask_stack, row["id"]
        else:
            return img_stack, row["id"]


def get_transforms(data="train"):
    """
    Get Albumentations transforms.
    Note: Resize/Padding is handled explicitly in Dataset,
    so we focus on spatial/pixel augmentations here.
    """
    if data == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
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
                ToTensorV2(),
            ]
        )
    return None
