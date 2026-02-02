import os
import json
import hashlib
import numpy as np
import pandas as pd
import rasterio
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from rasterio.windows import Window
from shapely.geometry import shape as shapely_shape, box
from shapely.ops import unary_union

from library.config import Config
from library.utils import rle_decode

# --- Helper Functions ---


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline based on the mode.
    """
    mean = Config.PIXEL_MEAN
    std = Config.PIXEL_STD

    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    ],
                    p=0.3,
                ),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3
                ),
                A.CLAHE(p=0.2),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_cortex_polygons(anatomical_json_path):
    """
    Parses the anatomical JSON to extract Cortex polygons using Shapely.
    Returns a unified Shapely polygon or None.
    """
    if not isinstance(anatomical_json_path, str) or not os.path.exists(
        anatomical_json_path
    ):
        return None

    try:
        with open(anatomical_json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    polygons = []
    for feature in data:
        # Check classification for 'Cortex'
        props = feature.get("properties", {})
        classification = props.get("classification", {})
        if classification.get("name") == "Cortex":
            geom = feature.get("geometry")
            if geom:
                try:
                    poly = shapely_shape(geom)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    polygons.append(poly)
                except Exception:
                    continue

    if not polygons:
        return None

    return unary_union(polygons)


def get_config_hash():
    """
    Generates a deterministic hash based on the configuration parameters.
    """
    config_dict = Config.get_config_dict()
    # Sort keys to ensure consistent ordering for hashing
    serialized = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    return hashlib.md5(serialized).hexdigest()


def ensure_masks_converted(df, mask_dir):
    """
    Ensures that RLE masks are converted to .npy files for fast random access.
    This is critical for memory efficiency during training.
    """
    os.makedirs(mask_dir, exist_ok=True)

    for _, row in df.iterrows():
        image_id = row["id"]
        npy_path = os.path.join(mask_dir, f"{image_id}.npy")

        if os.path.exists(npy_path):
            continue

        # If not exists, create it
        h, w = row["height_pixels"], row["width_pixels"]
        if "encoding" in row and pd.notna(row["encoding"]):
            mask = rle_decode(row["encoding"], (h, w))
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        np.save(npy_path, mask)


def generate_tile_coordinates(df, mode):
    """
    Generates a DataFrame of tile coordinates.
    For training, it calculates intersection with Cortex polygons to enable biased sampling.
    """
    tile_size = Config.TILE_SIZE
    # Use non-overlapping tiles for simplicity and speed, consistent with inference
    stride = Config.INFERENCE_STRIDE if mode != "train" else Config.TILE_SIZE

    coords_list = []

    for _, row in df.iterrows():
        image_id = row["id"]
        h, w = row["height_pixels"], row["width_pixels"]
        anatomical_path = row["anatomical_json_path"]

        # Get cortex polygon for this image if training
        cortex_poly = None
        if mode == "train":
            cortex_poly = get_cortex_polygons(anatomical_path)

        # Generate grid
        x_points = range(0, w, stride)
        y_points = range(0, h, stride)

        for y in y_points:
            for x in x_points:
                # Determine if this tile is within the Cortex
                is_cortex = False
                if mode == "train":
                    if cortex_poly:
                        tile_box = box(x, y, x + tile_size, y + tile_size)
                        if cortex_poly.intersects(tile_box):
                            is_cortex = True
                    else:
                        # If no anatomical info is available, assume it might be relevant
                        is_cortex = True

                coords_list.append(
                    {"id": image_id, "x": x, "y": y, "is_cortex": is_cortex}
                )

    return pd.DataFrame(coords_list)


def prepare_tiles(df, mode, load_cached_data=True):
    """
    Orchestrates tile generation with parameter-aware caching.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Generate hash based on current config
    config_hash = get_config_hash()
    cache_path = os.path.join(cache_dir, f"coords_{mode}_{config_hash}.parquet")

    # Handle masks caching (only for train/val sets that have encodings)
    if "encoding" in df.columns:
        mask_dir = os.path.join(Config.WORKING_DIR, "masks")
        ensure_masks_converted(df, mask_dir)

    # Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass  # Fallback to regenerate if cache is corrupt

    # Regenerate data
    coords_df = generate_tile_coordinates(df, mode)
    coords_df.to_parquet(cache_path, index=False)

    return coords_df


# --- Dataset Class ---


class HubmapDataset(Dataset):
    def __init__(self, metadata_df, mode="train", load_cached_data=True):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata containing file paths.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached tile coordinates.
        """
        self.mode = mode
        self.metadata_df = metadata_df.set_index("id")
        self.tile_size = Config.TILE_SIZE
        self.mask_dir = os.path.join(Config.WORKING_DIR, "masks")

        # Prepare tiles
        self.tiles_df = prepare_tiles(metadata_df, mode, load_cached_data)

        # Apply Cortex-Biased Sampling for Training
        if mode == "train":
            cortex_prob = Config.CORTEX_SAMPLING_PROB

            cortex_tiles = self.tiles_df[self.tiles_df["is_cortex"]]
            other_tiles = self.tiles_df[~self.tiles_df["is_cortex"]]

            # Balance the dataset based on CORTEX_SAMPLING_PROB
            if len(cortex_tiles) > 0:
                # Calculate how many 'other' tiles we need to satisfy the ratio
                # ratio = len(cortex) / (len(cortex) + len(other_sampled))
                # len(other_sampled) = len(cortex) * (1/ratio - 1)
                n_other = int(len(cortex_tiles) * (1 - cortex_prob) / cortex_prob)

                if n_other > len(other_tiles):
                    sampled_other = other_tiles
                else:
                    sampled_other = other_tiles.sample(
                        n=n_other, random_state=Config.SEED
                    )

                # Combine and shuffle
                self.tiles_df = (
                    pd.concat([cortex_tiles, sampled_other])
                    .sample(frac=1, random_state=Config.SEED)
                    .reset_index(drop=True)
                )
            else:
                # Fallback if no cortex tiles found (e.g. missing anatomical data)
                self.tiles_df = self.tiles_df.sample(
                    frac=1, random_state=Config.SEED
                ).reset_index(drop=True)

        # Limit for debugging
        if Config.DEBUG:
            self.tiles_df = self.tiles_df.head(Config.DEBUG_SAMPLES)

        self.transforms = get_transforms(mode)

    def __len__(self):
        return len(self.tiles_df)

    def __getitem__(self, idx):
        row = self.tiles_df.iloc[idx]
        image_id = row["id"]
        x, y = int(row["x"]), int(row["y"])

        # Get image path from metadata
        image_path = self.metadata_df.loc[image_id, "image_path"]

        # Read Image Window
        # Use rasterio for efficient windowed reading
        with rasterio.open(image_path) as src:
            # boundless=True automatically pads the image with fill_value if window extends beyond image
            window = Window(
                col_off=x, row_off=y, width=self.tile_size, height=self.tile_size
            )
            img = src.read(window=window, boundless=True, fill_value=0)

            # Cite debug_lesson_1: Dynamically Handle Variable Input Channels
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)

            # Rasterio reads as (C, H, W), move to (H, W, C) for Albumentations
            img = np.moveaxis(img, 0, -1)

        # Read Mask Window (if available)
        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)

        if self.mode in ["train", "val"]:
            npy_path = os.path.join(self.mask_dir, f"{image_id}.npy")
            if os.path.exists(npy_path):
                # Use mmap_mode='r' to avoid loading full file into RAM
                full_mask = np.load(npy_path, mmap_mode="r")

                h_full, w_full = full_mask.shape

                # Calculate slice indices with boundary checks
                x_start, y_start = x, y
                x_end, y_end = x + self.tile_size, y + self.tile_size

                x_start_clamped = max(0, x_start)
                y_start_clamped = max(0, y_start)
                x_end_clamped = min(w_full, x_end)
                y_end_clamped = min(h_full, y_end)

                if x_start_clamped < x_end_clamped and y_start_clamped < y_end_clamped:
                    mask_crop = full_mask[
                        y_start_clamped:y_end_clamped, x_start_clamped:x_end_clamped
                    ]

                    # Calculate offsets in the output mask
                    out_y = y_start_clamped - y_start
                    out_x = x_start_clamped - x_start

                    mask[
                        out_y : out_y + mask_crop.shape[0],
                        out_x : out_x + mask_crop.shape[1],
                    ] = mask_crop

        # Apply Transforms
        augmented = self.transforms(image=img, mask=mask)
        img_tensor = augmented["image"]
        mask_tensor = augmented["mask"]

        # Ensure mask has channel dimension (1, H, W)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "id": image_id,
            "x": x,
            "y": y,
        }
