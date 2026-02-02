import os
import cv2
import torch
import rasterio
import hashlib
import json
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from rasterio.windows import Window
from tqdm import tqdm

from library.config import Config
from library.utils import rle_decode, set_seed

# =========================================================================
# Mask Preprocessing
# =========================================================================


def preprocess_masks(df, mask_dir):
    """
    Decodes RLE masks and saves them as .npy files for fast access.
    """
    os.makedirs(mask_dir, exist_ok=True)

    # Filter only rows with encoding (train/val)
    if "encoding" not in df.columns:
        return

    print(f"Checking/Generating masks in {mask_dir}...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Masks"):
        image_id = row["id"]
        mask_path = os.path.join(mask_dir, f"{image_id}.npy")

        if os.path.exists(mask_path):
            continue

        # Decode RLE
        h, w = row["height_pixels"], row["width_pixels"]
        mask = rle_decode(row["encoding"], (h, w))

        # Save as binary numpy array (uint8)
        np.save(mask_path, mask)


# =========================================================================
# Coordinate Generation & Caching
# =========================================================================


def get_cache_path(prefix, params):
    """Generates a cache filename based on hashed parameters."""
    param_str = json.dumps(params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode("utf-8")).hexdigest()
    return os.path.join(Config.CACHE_DIR, f"{prefix}_{param_hash}.parquet")


def prepare_train_coordinates(df, mask_dir, load_cached_data=True):
    """
    Generates training coordinates with Explicit Positive Oversampling.
    """
    params = Config.get_params_dict()
    params["type"] = "train_coords"
    cache_path = get_cache_path("coords_train", params)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached training coordinates from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating training coordinates...")

    # 1. Generate pool of all possible coordinates
    all_coords = []
    tile_size = Config.TILE_SIZE

    # We use a stride smaller than tile_size to increase potential positive samples
    stride = tile_size // 2

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scanning Tiles"):
        image_id = row["id"]
        mask_path = os.path.join(mask_dir, f"{image_id}.npy")

        # Open mask in memory-mapped mode to avoid OOM
        try:
            mask = np.load(mask_path, mmap_mode="r")
        except FileNotFoundError:
            print(f"Warning: Mask not found for {image_id}, skipping.")
            continue

        h, w = mask.shape

        # Generate grid
        y_steps = range(0, h - tile_size + 1, stride)
        x_steps = range(0, w - tile_size + 1, stride)

        for y in y_steps:
            for x in x_steps:
                # Check if tile contains glomerulus
                # Slicing a memmap is fast
                tile_mask = mask[y : y + tile_size, x : x + tile_size]
                is_pos = np.any(tile_mask)

                all_coords.append({"id": image_id, "x": x, "y": y, "is_pos": is_pos})

    pool_df = pd.DataFrame(all_coords)

    if pool_df.empty:
        raise ValueError("No coordinates generated. Check mask paths and data.")

    # 2. Explicit Positive Oversampling
    pos_df = pool_df[pool_df["is_pos"] == True]
    neg_df = pool_df[pool_df["is_pos"] == False]

    n_samples = Config.TRAIN_NUM_SAMPLES
    n_pos = int(n_samples * Config.TRAIN_POS_RATIO)
    n_neg = n_samples - n_pos

    # Sample with replacement if not enough tiles
    sampled_pos = pos_df.sample(
        n=n_pos, replace=(len(pos_df) < n_pos), random_state=Config.SEED
    )
    sampled_neg = neg_df.sample(
        n=n_neg, replace=(len(neg_df) < n_neg), random_state=Config.SEED
    )

    final_df = (
        pd.concat([sampled_pos, sampled_neg])
        .sample(frac=1, random_state=Config.SEED)
        .reset_index(drop=True)
    )

    # Cache result
    final_df.to_parquet(cache_path)
    print(f"Saved {len(final_df)} training coordinates to {cache_path}")

    return final_df


def prepare_inference_coordinates(df, prefix="val", load_cached_data=True):
    """
    Generates dense grid coordinates for validation/testing (sliding window).
    """
    params = Config.get_params_dict()
    params["type"] = f"{prefix}_coords"
    # Remove random sampling params from hash for inference
    params.pop("train_num_samples", None)
    params.pop("train_pos_ratio", None)

    cache_path = get_cache_path(f"coords_{prefix}", params)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {prefix} coordinates from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Generating {prefix} coordinates...")

    coords = []
    tile_size = Config.TILE_SIZE

    # For inference, we tile the image.
    # We can use no overlap or small overlap.
    # Using no overlap for simplicity and speed in this implementation,
    # but handling edge cases by padding or shifting last tile.

    for _, row in df.iterrows():
        image_id = row["id"]
        h, w = row["height_pixels"], row["width_pixels"]

        # Simple grid
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Adjust if out of bounds (take the last crop ending at w/h)
                real_y = min(y, h - tile_size)
                real_x = min(x, w - tile_size)

                # Ensure we don't have negative coords if image < tile_size
                real_y = max(0, real_y)
                real_x = max(0, real_x)

                coords.append({"id": image_id, "x": real_x, "y": real_y})

    coords_df = pd.DataFrame(coords).drop_duplicates()

    coords_df.to_parquet(cache_path)
    print(f"Saved {len(coords_df)} {prefix} coordinates to {cache_path}")

    return coords_df


# =========================================================================
# Transforms
# =========================================================================


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.
    """
    mean = Config.NORM_MEAN
    std = Config.NORM_STD

    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Photometric Distortions
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=1.0
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=1.0,
                        ),
                    ],
                    p=0.5,
                ),
                A.CLAHE(p=0.2),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])


# =========================================================================
# Dataset
# =========================================================================


class HuBMAPDataset(Dataset):
    def __init__(self, df, coords_df, mask_dir=None, transforms=None, phase="train"):
        self.df = df.set_index("id")
        self.coords_df = coords_df
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.phase = phase
        self.tile_size = Config.TILE_SIZE

    def __len__(self):
        return len(self.coords_df)

    def __getitem__(self, idx):
        row = self.coords_df.iloc[idx]
        image_id = row["id"]
        x, y = int(row["x"]), int(row["y"])

        # Get image path
        img_path = self.df.loc[image_id, "image_path"]

        # Read Image Tile
        # Use rasterio Window to read only the specific crop
        with rasterio.open(img_path) as src:
            # Rasterio expects (col_off, row_off, width, height)
            window = Window(x, y, self.tile_size, self.tile_size)

            # Read and transpose to (H, W, C)
            img = src.read(window=window)

            # Handle edge case where window might be smaller than tile_size (if image is small)
            # Though our coordinate generation logic tries to avoid this by shifting.
            # If shape mismatch, pad.
            if img.shape[1] != self.tile_size or img.shape[2] != self.tile_size:
                pad_h = self.tile_size - img.shape[1]
                pad_w = self.tile_size - img.shape[2]
                img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")

            img = np.moveaxis(img, 0, -1)  # (C, H, W) -> (H, W, C)

        # Read Mask Tile (if train/val)
        mask = None
        if self.phase in ["train", "val"]:
            mask_path = os.path.join(self.mask_dir, f"{image_id}.npy")
            # Load using mmap to save RAM
            full_mask = np.load(mask_path, mmap_mode="r")
            mask = full_mask[y : y + self.tile_size, x : x + self.tile_size]

            # Pad if needed
            if mask.shape[0] != self.tile_size or mask.shape[1] != self.tile_size:
                pad_h = self.tile_size - mask.shape[0]
                pad_w = self.tile_size - mask.shape[1]
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")

            mask = mask.astype(np.float32)

        # Apply Transforms
        if self.transforms:
            if mask is not None:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
                # Ensure mask has channel dimension
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
            else:
                augmented = self.transforms(image=img)
                img = augmented["image"]

        if self.phase == "test":
            return img, image_id, x, y

        return img, mask


# =========================================================================
# Data Loaders
# =========================================================================


def get_dataloaders(train_df=None, val_df=None, test_df=None):
    """
    Prepares DataLoaders for train, val, and test.
    Handles mask preprocessing and coordinate caching.
    """
    Config.setup()
    mask_dir = os.path.join(Config.WORKING_DIR, "masks")

    dataloaders = {}

    # --- Training Setup ---
    if train_df is not None:
        # 1. Preprocess masks (convert RLE to npy)
        preprocess_masks(train_df, mask_dir)

        # 2. Generate/Load Coordinates
        train_coords = prepare_train_coordinates(train_df, mask_dir)

        # 3. Dataset & Loader
        train_ds = HuBMAPDataset(
            train_df,
            train_coords,
            mask_dir=mask_dir,
            transforms=get_transforms("train"),
            phase="train",
        )

        dataloaders["train"] = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

    # --- Validation Setup ---
    if val_df is not None:
        # Ensure masks exist for val set too
        preprocess_masks(val_df, mask_dir)

        val_coords = prepare_inference_coordinates(val_df, prefix="val")

        val_ds = HuBMAPDataset(
            val_df,
            val_coords,
            mask_dir=mask_dir,
            transforms=get_transforms("val"),
            phase="val",
        )

        dataloaders["val"] = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    # --- Test Setup ---
    if test_df is not None:
        test_coords = prepare_inference_coordinates(test_df, prefix="test")

        test_ds = HuBMAPDataset(
            test_df,
            test_coords,
            mask_dir=None,  # No masks for test
            transforms=get_transforms("test"),
            phase="test",
        )

        dataloaders["test"] = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    return dataloaders
