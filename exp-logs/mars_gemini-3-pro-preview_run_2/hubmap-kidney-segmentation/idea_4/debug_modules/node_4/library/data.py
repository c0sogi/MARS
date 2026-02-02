import os
import cv2
import json
import torch
import rasterio
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from rasterio.windows import Window
from library.config import Config

# ====================================================
# Helper Functions
# ====================================================


def get_transforms(mode="train"):
    """
    Returns the Albumentations composition for training or validation.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return A.Compose(
            [
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                # Color Augmentations (Crucial for staining variations)
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Normalization and Tensor Conversion
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


def load_polygons(json_path, classification_filter=None):
    """
    Parses a HuBMAP JSON file and returns a list of polygons.

    Args:
        json_path (str): Path to the JSON file.
        classification_filter (str, optional): If provided, only returns polygons
                                               with this classification name (e.g., "Cortex").

    Returns:
        list: A list of numpy arrays, where each array is a polygon of shape (N, 2).
    """
    if not os.path.exists(json_path):
        return []

    with open(json_path, "r") as f:
        data = json.load(f)

    polygons = []
    for feature in data:
        # Check geometry type
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "Polygon":
            continue

        # Check classification if filter is provided
        if classification_filter:
            props = feature.get("properties", {})
            classification = props.get("classification", {})
            name = classification.get("name")
            if name != classification_filter:
                continue

        coordinates = geometry.get("coordinates", [])
        if not coordinates:
            continue

        # Coordinates are usually a list of lists of points: [[x, y], [x, y], ...]
        # Sometimes nested: [[[x, y], ...]]
        pts = np.array(coordinates[0], dtype=np.int32)
        polygons.append(pts)

    return polygons


def prepare_tiles(df, mode="train", load_cached_data=True):
    """
    Generates a list of tile coordinates for the dataset.
    Implements Representative Undersampling for the training set.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        list: List of dictionaries containing tile metadata.
    """
    cache_filename = f"{mode}_tiles.npy"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached tiles from {cache_path}...")
        try:
            tiles = np.load(cache_path, allow_pickle=True).tolist()
            return tiles
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    print(f"Generating tiles for {mode} set...")
    tiles = []

    # Define stride
    # For training, we use non-overlapping tiles (stride = TILE_SIZE) to keep it simple and balanced.
    # For validation/test, we might want overlap, but here we define the grid for the Dataset.
    # Inference overlap is usually handled in the inference loop, not the dataset class.
    stride = Config.TILE_SIZE

    for idx, row in df.iterrows():
        img_id = row["id"]
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        json_path = os.path.join(Config.INPUT_DIR, row["json_path"])

        # Read image dimensions
        with rasterio.open(img_path) as src:
            h, w = src.height, src.width

        # If training, we need to know which tiles have glomeruli for undersampling
        mask = None
        if mode == "train":
            # Load Glomerulus polygons
            glom_polys = load_polygons(json_path)
            # Create a temporary low-res mask or check intersection?
            # Given memory constraints, drawing full mask is risky if image is huge (50k x 50k).
            # However, 50k x 50k uint8 is 2.5GB. We have 220GB RAM. It is safe.
            mask = np.zeros((h, w), dtype=np.uint8)
            if glom_polys:
                cv2.fillPoly(mask, glom_polys, 1)

        # Generate grid
        # x = col, y = row
        x_points = range(0, w, stride)
        y_points = range(0, h, stride)

        img_tiles = []

        for y in y_points:
            for x in x_points:
                # Adjust size for edges
                # We will pad in __getitem__, but here we just record the top-left

                has_glom = False
                if mode == "train":
                    # Check if this tile contains any glomerulus pixels
                    # Slice carefully to avoid out of bounds
                    y_end = min(y + Config.TILE_SIZE, h)
                    x_end = min(x + Config.TILE_SIZE, w)
                    if mask[y:y_end, x:x_end].sum() > 0:
                        has_glom = True

                tile_meta = {
                    "id": img_id,
                    "image_path": row["image_path"],
                    "json_path": row["json_path"],
                    "anatomical_json_path": row["anatomical_json_path"],
                    "x": x,
                    "y": y,
                    "h": h,  # Store original image dims
                    "w": w,
                    "has_glom": has_glom,
                }
                img_tiles.append(tile_meta)

        # Undersampling Logic (Per Image or Global? Global is better, but per image ensures coverage)
        # Let's collect all tiles first, then filter globally for the dataset.
        tiles.extend(img_tiles)

        # Clean up memory
        del mask

    # Apply Representative Undersampling for Training
    if mode == "train":
        pos_tiles = [t for t in tiles if t["has_glom"]]
        neg_tiles = [t for t in tiles if not t["has_glom"]]

        # Sample negative tiles
        num_neg = int(len(neg_tiles) * Config.NEGATIVE_SAMPLE_RATE)
        # Use a fixed seed for reproducibility of the split
        np.random.seed(Config.SEED)
        if num_neg > 0:
            neg_tiles_sampled = np.random.choice(
                neg_tiles, num_neg, replace=False
            ).tolist()
        else:
            neg_tiles_sampled = []

        final_tiles = pos_tiles + neg_tiles_sampled
        np.random.shuffle(final_tiles)
        print(f"  Total tiles: {len(tiles)}")
        print(f"  Positive tiles: {len(pos_tiles)}")
        print(f"  Negative tiles (Sampled): {len(neg_tiles_sampled)}")
        print(f"  Final Training Set Size: {len(final_tiles)}")
        tiles = final_tiles
    else:
        print(f"  {mode.capitalize()} Set Size: {len(tiles)} tiles")

    # Cache the result
    try:
        np.save(cache_path, np.array(tiles))
        print(f"Saved tiles to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    return tiles


# ====================================================
# Dataset Class
# ====================================================


class HuBMAPDataset(Dataset):
    def __init__(self, tiles, transforms=None, mode="train"):
        """
        Args:
            tiles (list): List of tile metadata dictionaries.
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.tiles = tiles
        self.transforms = transforms
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

        # Pre-load polygons to avoid parsing JSON in the loop
        # We group polygons by image ID
        self.polygons = {}

        # Identify unique images in this dataset
        unique_ids = set(t["id"] for t in self.tiles)

        # We also need the mapping from ID to paths.
        # We can get this from the first tile of each ID.
        id_to_paths = {}
        for t in self.tiles:
            if t["id"] not in id_to_paths:
                id_to_paths[t["id"]] = (t["json_path"], t["anatomical_json_path"])

        if self.mode in ["train", "val"]:
            print("Pre-loading polygons for mask generation...")
            for img_id in unique_ids:
                json_path, anat_json_path = id_to_paths[img_id]

                # Load Glomerulus Polygons
                glom_polys = load_polygons(os.path.join(self.input_dir, json_path))

                # Load Cortex Polygons (Auxiliary Target)
                # We filter for "Cortex"
                cortex_polys = load_polygons(
                    os.path.join(self.input_dir, anat_json_path),
                    classification_filter="Cortex",
                )

                self.polygons[img_id] = {"glom": glom_polys, "cortex": cortex_polys}

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile_info = self.tiles[idx]
        img_id = tile_info["id"]
        x, y = tile_info["x"], tile_info["y"]

        # 1. Load Image Tile
        img_path = os.path.join(self.input_dir, tile_info["image_path"])

        # Calculate window
        # Handle edge cases where tile extends beyond image
        # rasterio handles reading, but we need to pad result if smaller
        w_read = min(Config.TILE_SIZE, tile_info["w"] - x)
        h_read = min(Config.TILE_SIZE, tile_info["h"] - y)

        with rasterio.open(img_path) as src:
            # Read window: (bands, height, width)
            # rasterio uses (col_off, row_off, width, height)
            window = Window(x, y, w_read, h_read)
            img = src.read(window=window)

        # Ensure 3 channels (Handle grayscale or >3 channel images)
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        elif img.shape[0] > 3:
            img = img[:3, :, :]

        # Convert to (H, W, C) and RGB
        img = np.moveaxis(img, 0, -1)

        # Handle Padding if tile is smaller than TILE_SIZE (at edges)
        if h_read < Config.TILE_SIZE or w_read < Config.TILE_SIZE:
            pad_h = Config.TILE_SIZE - h_read
            pad_w = Config.TILE_SIZE - w_read
            # Pad with reflection or zeros. Reflection is better for CNNs.
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        # 2. Generate Masks (Train/Val only)
        mask = None
        if self.mode in ["train", "val"]:
            # Initialize 2-channel mask: [H, W, 2]
            # Channel 0: Glom, Channel 1: Cortex
            mask = np.zeros((Config.TILE_SIZE, Config.TILE_SIZE, 2), dtype=np.float32)

            polys = self.polygons[img_id]

            # Draw Glomeruli (Channel 0)
            if polys["glom"]:
                # Shift polygon coordinates relative to tile
                # Filter polygons that are likely within the tile to speed up?
                # cv2.fillPoly is fast enough to just try drawing shifted polys.
                # It clips automatically.
                shifted_gloms = [p - [x, y] for p in polys["glom"]]

                # Draw on a temporary single channel
                glom_mask = np.zeros(
                    (Config.TILE_SIZE, Config.TILE_SIZE), dtype=np.uint8
                )
                cv2.fillPoly(glom_mask, shifted_gloms, 1)
                mask[:, :, 0] = glom_mask

            # Draw Cortex (Channel 1)
            if polys["cortex"]:
                shifted_cortex = [p - [x, y] for p in polys["cortex"]]
                cortex_mask = np.zeros(
                    (Config.TILE_SIZE, Config.TILE_SIZE), dtype=np.uint8
                )
                cv2.fillPoly(cortex_mask, shifted_cortex, 1)
                mask[:, :, 1] = cortex_mask

        # 3. Augmentations
        if self.transforms:
            if mask is not None:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
                # Mask comes out as (H, W, C) from albumentations if multi-channel?
                # ToTensorV2 converts image to (C, H, W) but mask behavior depends.
                # Usually ToTensorV2 converts mask to (H, W) if single channel or (H, W, C) -> (C, H, W).
                # Let's ensure mask is (C, H, W) float
                if mask.ndim == 3 and mask.shape[0] != 2:  # if (H, W, 2)
                    mask = mask.permute(2, 0, 1)
            else:
                augmented = self.transforms(image=img)
                img = augmented["image"]

        # Return
        if self.mode in ["train", "val"]:
            return img, mask
        else:
            # For test, we might need metadata to reconstruct the image
            return img, torch.tensor([x, y])
