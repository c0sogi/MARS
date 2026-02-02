import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode


def load_image(rel_path):
    """
    Loads an image from the input directory.
    Handles 16-bit to float conversion if necessary.
    """
    full_path = os.path.join(Config.INPUT_DIR, rel_path)
    if not os.path.exists(full_path):
        # Should not happen given metadata validation, but safe fallback
        return np.zeros((256, 256), dtype=np.float32)

    # Load unchanged to preserve bit-depth (likely 16-bit or 8-bit)
    img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return np.zeros((256, 256), dtype=np.float32)

    # Ensure float32 for processing
    img = img.astype(np.float32)
    return img


def robust_normalize(img):
    """
    Applies Robust Min-Max Normalization based on 1st and 99th percentiles.
    Operates per-channel.
    """
    # Ensure (H, W, C)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        p1 = np.percentile(channel, 1)
        p99 = np.percentile(channel, 99)

        # Avoid division by zero for constant images
        if p99 - p1 < 1e-6:
            out[:, :, c] = 0
        else:
            clipped = np.clip(channel, p1, p99)
            out[:, :, c] = (clipped - p1) / (p99 - p1)

    return out


def resize_pad_to_square(img, target_size, mask=None):
    """
    Resizes the image to fit within target_size while preserving aspect ratio,
    then pads the shorter dimension to create a square image.

    Args:
        img: (H, W, C) Image array
        target_size: (H_out, W_out) Tuple
        mask: (H, W, C) Mask array (optional)

    Returns:
        img_padded, mask_padded, padding_info
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size

    # 1. Scale to fit longest dimension
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    mask_resized = None
    if mask is not None:
        # Use Nearest Neighbor for masks to preserve binary values
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

    # 2. Pad to target size (Center Padding)
    pad_h = target_h - new_h
    pad_w = target_w - new_w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Apply padding
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )

    mask_padded = None
    if mask is not None:
        mask_padded = cv2.copyMakeBorder(
            mask_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )

    # Metadata for inverse transform
    padding_info = {
        "original_height": h,
        "original_width": w,
        "new_height": new_h,
        "new_width": new_w,
        "pad_top": top,
        "pad_left": left,
        "scale": scale,
    }

    return img_padded, mask_padded, padding_info


def process_metadata(
    csv_path, load_cached_data=True, cache_name="metadata_25d.parquet"
):
    """
    Loads metadata and prepares 2.5D slice paths (prev, curr, next).
    Caches the result to disk.
    """
    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached metadata from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Process from Scratch
    # print(f"Processing metadata from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Ensure sorting by Case -> Day -> Slice
    # Slice numbers in filenames are usually zero-padded strings, but safe to cast
    df["slice_int"] = df["slice"].astype(int)
    df = df.sort_values(["case", "day", "slice_int"]).reset_index(drop=True)

    # Create grouping key
    df["case_day"] = df["case"].astype(str) + "_" + df["day"].astype(str)

    # Identify neighbors
    # Shift paths
    df["image_path_prev"] = df["image_path"].shift(1)
    df["image_path_next"] = df["image_path"].shift(-1)

    # Shift keys to check boundaries
    df["case_day_prev"] = df["case_day"].shift(1)
    df["case_day_next"] = df["case_day"].shift(-1)

    # Boundary Logic: If neighbor is different case/day, use current slice
    # Previous
    mask_prev = df["case_day"] != df["case_day_prev"]
    df.loc[mask_prev, "image_path_prev"] = df.loc[mask_prev, "image_path"]

    # Next
    mask_next = df["case_day"] != df["case_day_next"]
    df.loc[mask_next, "image_path_next"] = df.loc[mask_next, "image_path"]

    # Handle NaN at very end of dataframe (shift -1)
    df["image_path_next"] = df["image_path_next"].fillna(df["image_path"])
    # Handle NaN at very start (shift 1) - covered by mask_prev usually, but for safety
    df["image_path_prev"] = df["image_path_prev"].fillna(df["image_path"])

    # Cleanup
    drop_cols = ["slice_int", "case_day", "case_day_prev", "case_day_next"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 3. Save Cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path)
    # print(f"Saved processed metadata to {cache_path}")

    return df


class UWGI_25D_Dataset(Dataset):
    """
    Dataset class for 2.5D segmentation.
    Loads 3 slices (t-1, t, t+1) as channels.
    """

    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- 1. Load Images (2.5D Stack) ---
        # Paths determined by process_metadata
        paths = [row["image_path_prev"], row["image_path"], row["image_path_next"]]
        images = []
        for p in paths:
            img = load_image(p)
            images.append(img)

        # Stack to (H, W, 3)
        img_stack = np.stack(images, axis=-1)

        # --- 2. Load Mask (Train/Val only) ---
        mask = None
        if self.mode in ["train", "val"]:
            h_orig, w_orig = row["height"], row["width"]
            mask = np.zeros((h_orig, w_orig, Config.NUM_CLASSES), dtype=np.float32)

            for i, cls in enumerate(Config.CLASSES):
                rle = row[cls]
                if pd.notna(rle) and rle != "":
                    mask[:, :, i] = rle_decode(rle, (h_orig, w_orig))

        # --- 3. Preprocessing Pipeline ---

        # A. Robust Normalization (Intensity)
        # Done before resizing to calculate percentiles on original pixel distribution
        img_stack = robust_normalize(img_stack)

        # B. Geometry-Preserving Resize & Pad (Spatial)
        img_proc, mask_proc, pad_info = resize_pad_to_square(
            img_stack, Config.IMG_SIZE, mask
        )

        # C. Augmentations (Spatial/Intensity)
        if self.transforms:
            if self.mode in ["train", "val"]:
                transformed = self.transforms(image=img_proc, mask=mask_proc)
                img_proc = transformed["image"]
                mask_proc = transformed["mask"]
            else:
                transformed = self.transforms(image=img_proc)
                img_proc = transformed["image"]

        # --- 4. Format Output ---
        # Ensure tensors (Albumentations ToTensorV2 usually handles this, but safety check)
        if not isinstance(img_proc, torch.Tensor):
            img_proc = torch.from_numpy(img_proc).permute(2, 0, 1).float()

        if mask_proc is not None and not isinstance(mask_proc, torch.Tensor):
            mask_proc = torch.from_numpy(mask_proc).permute(2, 0, 1).float()

        result = {
            "image": img_proc,
            "id": row["id"],
            "padding_info": pad_info,  # Crucial for coordinate correction
        }

        if mask_proc is not None:
            result["mask"] = mask_proc

        return result


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    Note: Resizing and Normalization are handled manually in the Dataset class
    to support the specific Geometry-Preserving and Robust Norm requirements.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    ],
                    p=0.3,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_dataloaders(config):
    """
    Creates DataLoaders for Train, Val, and Test.
    """
    # --- Train ---
    train_df = process_metadata(
        config.TRAIN_DF_PATH,
        load_cached_data=True,
        cache_name="train_metadata_25d.parquet",
    )

    if config.DEBUG:
        train_df = train_df.sample(n=300, random_state=config.SEED).reset_index(
            drop=True
        )

    train_ds = UWGI_25D_Dataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # --- Validation ---
    val_df = process_metadata(
        config.VAL_DF_PATH, load_cached_data=True, cache_name="val_metadata_25d.parquet"
    )

    if config.DEBUG:
        val_df = val_df.sample(n=100, random_state=config.SEED).reset_index(drop=True)

    val_ds = UWGI_25D_Dataset(val_df, transforms=get_transforms("val"), mode="val")

    val_loader = DataLoader(
        val_ds,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Test ---
    test_df = process_metadata(
        config.TEST_DF_PATH,
        load_cached_data=True,
        cache_name="test_metadata_25d.parquet",
    )

    test_ds = UWGI_25D_Dataset(test_df, transforms=get_transforms("test"), mode="test")

    test_loader = DataLoader(
        test_ds,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
