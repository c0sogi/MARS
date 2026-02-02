import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PREFETCH_FACTOR,
    SEED,
)
from library.utils import rle_decode


def prepare_data(csv_path, mode="train", load_cached_data=True):
    """
    Prepares the dataframe for the dataset.
    Adds columns for previous and next slice paths for 2.5D input.
    Implements caching using parquet.
    """
    # Define cache path
    filename = os.path.basename(csv_path).replace(".csv", "")
    cache_path = os.path.join(WORKING_DIR, f"{filename}_25d_processed.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    df = pd.read_csv(csv_path)

    # Ensure slice is integer for sorting
    df["slice"] = df["slice"].astype(int)

    # Sort to ensure correct ordering for neighbor identification
    df = df.sort_values(["case", "day", "slice"]).reset_index(drop=True)

    # Generate paths for t-1, t, t+1
    # We create shifted versions of the dataframe to align neighbors
    # We must ensure we don't cross case/day boundaries

    # Group identifier
    df["group_id"] = df["case"].astype(str) + "_" + df["day"].astype(str)

    # Get image paths
    paths = df["image_path"].values
    group_ids = df["group_id"].values

    # Vectorized neighbor logic
    n = len(df)

    # Previous slice (t-1)
    # Default to current path
    paths_prev = paths.copy()
    # Indices where previous row is same group
    prev_indices = np.arange(n) - 1
    valid_prev = (prev_indices >= 0) & (group_ids == np.roll(group_ids, 1))
    # Where valid, use previous path. Where invalid (start of group), keep current path (replicate)
    paths_prev[valid_prev] = paths[prev_indices[valid_prev]]

    # Next slice (t+1)
    paths_next = paths.copy()
    next_indices = np.arange(n) + 1
    valid_next = (next_indices < n) & (group_ids == np.roll(group_ids, -1))
    paths_next[valid_next] = paths[next_indices[valid_next]]

    df["image_path_prev"] = paths_prev
    df["image_path_next"] = paths_next

    # Drop temp column
    df = df.drop(columns=["group_id"])

    # Fill NaNs in mask columns if they exist (for train/val)
    if mode in ["train", "val"]:
        for col in ["large_bowel", "small_bowel", "stomach"]:
            if col in df.columns:
                df[col] = df[col].fillna("")

    # 3. Save to cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class UWDataset(Dataset):
    def __init__(self, df, mode="train", transform=None):
        self.df = df
        self.mode = mode
        self.transform = transform

        # Pre-extract columns to arrays for faster access
        self.image_paths = df["image_path"].values
        self.prev_paths = df["image_path_prev"].values
        self.next_paths = df["image_path_next"].values

        # Metadata for resizing masks correctly
        self.heights = df["height"].values
        self.widths = df["width"].values

        if self.mode != "test":
            self.large_bowel = df["large_bowel"].values
            self.small_bowel = df["small_bowel"].values
            self.stomach = df["stomach"].values

    def __len__(self):
        return len(self.df)

    def normalize_slice(self, img):
        """
        Applies robust per-slice normalization:
        1. Clip to [p1, p99]
        2. Min-Max scale to [0, 1]
        """
        img = img.astype(np.float32)

        # Avoid calculating percentiles on empty/black images to save time and avoid errors
        if img.max() == 0:
            return img

        p1 = np.percentile(img, 1)
        p99 = np.percentile(img, 99)

        img = np.clip(img, p1, p99)

        min_val = img.min()
        max_val = img.max()

        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img[:] = 0

        return img

    def load_image_volume(self, index):
        """
        Loads t-1, t, t+1 images and stacks them.
        """
        paths = [
            self.prev_paths[index],
            self.image_paths[index],
            self.next_paths[index],
        ]
        images = []

        for p in paths:
            full_path = os.path.join(INPUT_DIR, p)
            # Load 16-bit or 8-bit unchanged
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

            # Handle case where image might fail to load
            if img is None:
                # Create a dummy black image of expected size if load fails
                # Use metadata dimensions
                h, w = self.heights[index], self.widths[index]
                img = np.zeros((h, w), dtype=np.float32)

            # If image has multiple channels (e.g. saved as RGB), take the first one
            if len(img.shape) > 2:
                img = img[..., 0]

            # Normalize immediately
            img = self.normalize_slice(img)
            images.append(img)

        # Stack along channel dimension: (H, W, 3)
        return np.stack(images, axis=-1)

    def load_masks(self, index, h, w):
        """
        Decodes RLE masks for the 3 classes.
        """
        masks = np.zeros((h, w, 3), dtype=np.float32)

        # Class 0: Large Bowel
        rle_lb = self.large_bowel[index]
        masks[..., 0] = rle_decode(rle_lb, shape=(h, w))

        # Class 1: Small Bowel
        rle_sb = self.small_bowel[index]
        masks[..., 1] = rle_decode(rle_sb, shape=(h, w))

        # Class 2: Stomach
        rle_stomach = self.stomach[index]
        masks[..., 2] = rle_decode(rle_stomach, shape=(h, w))

        return masks

    def __getitem__(self, index):
        # 1. Load 2.5D Image Volume
        image = self.load_image_volume(index)

        # 2. Load Masks (if not test)
        if self.mode != "test":
            h, w = self.heights[index], self.widths[index]
            mask = self.load_masks(index, h, w)

            # 3. Augmentations
            if self.transform:
                data = self.transform(image=image, mask=mask)
                image = data["image"]
                mask = data["mask"]

            # Permute mask to (C, H, W) if it's a tensor (ToTensorV2 does this for image but not always mask depending on config)
            # Albumentations ToTensorV2 converts image to (C, H, W) and mask to (H, W, C) usually, or (H, W) if single channel.
            # Here mask is (H, W, 3). ToTensorV2 will make it (3, H, W) if transpose_mask=True (default False) or we do it manually.
            # Standard ToTensorV2 converts image to tensor (C, H, W). Mask is converted to tensor but shape depends.
            # We will manually ensure shape (C, H, W).

            if isinstance(mask, torch.Tensor):
                if mask.shape[-1] == 3:  # (H, W, C)
                    mask = mask.permute(2, 0, 1)
            else:
                # If transform didn't convert to tensor (unlikely with ToTensorV2)
                mask = mask.transpose(2, 0, 1)
                mask = torch.from_numpy(mask)

            return image, mask

        else:
            # Test mode
            if self.transform:
                data = self.transform(image=image)
                image = data["image"]

            # Return dummy mask or just image. Returning dummy mask keeps collate_fn simple.
            return image, torch.zeros((3, image.shape[1], image.shape[2]))


def get_transforms(mode="train"):
    """
    Returns albumentations transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(
                    height=IMAGE_SIZE[0],
                    width=IMAGE_SIZE[1],
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
    else:
        # Val and Test
        return A.Compose(
            [
                A.Resize(
                    height=IMAGE_SIZE[0],
                    width=IMAGE_SIZE[1],
                    interpolation=cv2.INTER_LINEAR,
                ),
                ToTensorV2(),
            ]
        )


def get_loaders(load_cached_data=True, debug=False):
    """
    Prepares DataLoaders for training and validation.

    Args:
        load_cached_data (bool): Whether to use cached parquet files.
        debug (bool): If True, subsamples the dataset for quick debugging.
    """
    # 1. Prepare Dataframes
    train_df = prepare_data(TRAIN_CSV, mode="train", load_cached_data=load_cached_data)
    val_df = prepare_data(VAL_CSV, mode="val", load_cached_data=load_cached_data)

    if debug:
        train_df = train_df.iloc[: BATCH_SIZE * 2]
        val_df = val_df.iloc[: BATCH_SIZE * 2]

    # 2. Create Datasets
    train_dataset = UWDataset(
        train_df, mode="train", transform=get_transforms(mode="train")
    )

    val_dataset = UWDataset(val_df, mode="val", transform=get_transforms(mode="val"))

    # 3. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=PREFETCH_FACTOR,
    )

    return train_loader, val_loader
