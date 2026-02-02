import os
import json
import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded string into a binary mask.
    """
    if pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def rasterize_anatomical_json(json_path, shape):
    """
    Rasterizes anatomical structure JSON into a binary mask.
    Cortex polygons are filled with 1, others (Medulla) left as 0.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if not os.path.exists(json_path):
        return mask

    with open(json_path, "r") as f:
        data = json.load(f)

    for feature in data:
        # Check properties for classification
        props = feature.get("properties", {})
        classification = props.get("classification", {})
        name = classification.get("name", "")

        # We only create a mask for the Cortex
        if name == "Cortex":
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [])

            for poly in coords:
                # poly is a list of [x, y] coordinates
                pts = np.array(poly, dtype=np.int32)
                # cv2.fillPoly expects a list of arrays
                cv2.fillPoly(mask, [pts], 1)

    return mask


class HuBMAPDataset(Dataset):
    """
    Dataset class for HuBMAP FTU detection.
    Handles 4-channel input (RGB + Anatomical Mask).
    """

    def __init__(self, tiles, masks=None, transform=None, color_transform=None):
        self.tiles = tiles
        self.masks = masks
        self.transform = transform
        self.color_transform = color_transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        # image shape: (H, W, 4)
        image = self.tiles[idx]

        # Separate RGB and Anatomy for specific augmentations
        rgb = image[:, :, :3]
        anat = image[:, :, 3]

        if self.masks is not None:
            mask = self.masks[idx]
        else:
            mask = np.zeros(image.shape[:2], dtype=np.float32)

        # Apply Color Augmentations to RGB only (Training only)
        if self.color_transform:
            augmented_rgb = self.color_transform(image=rgb)["image"]
            rgb = augmented_rgb

        # Recombine for Geometric Augmentations (applied to both image and mask)
        image_combined = np.dstack([rgb, anat])

        if self.transform:
            augmented = self.transform(image=image_combined, mask=mask)
            image_combined = augmented["image"]
            mask = augmented["mask"]

        # Manual Normalization and Tensor Conversion
        # Handle RGB Normalization (ImageNet stats)
        if isinstance(image_combined, torch.Tensor):
            image_combined = image_combined.numpy()

        # Ensure float32
        image_combined = image_combined.astype(np.float32)

        # RGB: [0, 255] -> [0, 1] -> Normalize
        rgb_norm = image_combined[:, :, :3] / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_norm = (rgb_norm - mean) / std

        # Anatomy: Keep as 0/1 (already uint8 0/1, just cast to float)
        # No normalization needed for binary mask channel, just scaling if needed.
        # Here we keep it as 0.0 or 1.0.
        anat_norm = image_combined[:, :, 3:4]

        # Concatenate back
        final_img = np.concatenate([rgb_norm, anat_norm], axis=2)

        # Convert to Tensor (C, H, W)
        final_img = torch.from_numpy(final_img.transpose(2, 0, 1)).float()

        # Mask to Tensor (1, H, W)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return final_img, mask


class TestDataset(Dataset):
    """
    Dataset for Inference.
    Yields tiles, coordinates, and original image info for reconstruction.
    """

    def __init__(self, df):
        self.df = df
        self.tile_size = Config.TILE_SIZE
        self.stride = Config.STRIDE
        self.data = self._prepare_tiles()

    def _prepare_tiles(self):
        data = []
        for _, row in self.df.iterrows():
            img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
            json_path = os.path.join(Config.INPUT_DIR, row["anatomical_json_path"])
            img_id = row["id"]

            # Load Image
            with rasterio.open(img_path) as src:
                image = src.read()  # (C, H, W)
                image = np.moveaxis(image, 0, -1)  # (H, W, C)

            h, w = image.shape[:2]

            # Load Anatomy
            anat_mask = rasterize_anatomical_json(json_path, (h, w))
            image_4ch = np.dstack([image, anat_mask])

            # Generate coordinates
            y_positions = list(range(0, h - self.tile_size + 1, self.stride))
            x_positions = list(range(0, w - self.tile_size + 1, self.stride))

            if (h - self.tile_size) % self.stride != 0:
                y_positions.append(h - self.tile_size)
            if (w - self.tile_size) % self.stride != 0:
                x_positions.append(w - self.tile_size)

            y_positions = sorted(list(set(y_positions)))
            x_positions = sorted(list(set(x_positions)))

            for y in y_positions:
                for x in x_positions:
                    tile = image_4ch[y : y + self.tile_size, x : x + self.tile_size]
                    data.append(
                        {
                            "image": tile,
                            "coords": np.array([y, x]),
                            "id": img_id,
                            "shape": np.array([h, w]),
                        }
                    )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"].astype(np.float32)

        # Normalize
        rgb_norm = image[:, :, :3] / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_norm = (rgb_norm - mean) / std

        anat_norm = image[:, :, 3:4]

        final_img = np.concatenate([rgb_norm, anat_norm], axis=2)
        final_img = torch.from_numpy(final_img.transpose(2, 0, 1)).float()

        return final_img, item["coords"], item["id"], item["shape"]


def process_and_cache_data(
    metadata_path, cache_path_tiles, cache_path_masks, is_train=True
):
    """
    Loads raw images, tiles them, applies representative undersampling (if train),
    and caches the result to disk.
    """
    df = pd.read_csv(metadata_path)

    all_tiles = []
    all_masks = []

    tile_size = Config.TILE_SIZE
    # Use overlap for training to increase data, same for validation to ensure coverage
    stride = Config.STRIDE

    for _, row in df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        json_path = os.path.join(Config.INPUT_DIR, row["anatomical_json_path"])

        # Load Image
        with rasterio.open(img_path) as src:
            image = src.read()
            image = np.moveaxis(image, 0, -1)

        h, w = image.shape[:2]

        # Load Anatomy
        anat_mask = rasterize_anatomical_json(json_path, (h, w))

        # Load GT Mask
        if "encoding" in row and pd.notna(row["encoding"]):
            gt_mask = rle_decode(row["encoding"], (h, w))
        else:
            gt_mask = np.zeros((h, w), dtype=np.uint8)

        # Fuse
        image_4ch = np.dstack([image, anat_mask])

        # Tiling
        y_positions = list(range(0, h - tile_size + 1, stride))
        x_positions = list(range(0, w - tile_size + 1, stride))

        if (h - tile_size) % stride != 0:
            y_positions.append(h - tile_size)
        if (w - tile_size) % stride != 0:
            x_positions.append(w - tile_size)

        y_positions = sorted(list(set(y_positions)))
        x_positions = sorted(list(set(x_positions)))

        for y in y_positions:
            for x in x_positions:
                tile_img = image_4ch[y : y + tile_size, x : x + tile_size]
                tile_mask = gt_mask[y : y + tile_size, x : x + tile_size]

                # Skip completely empty image regions (padding artifacts)
                if tile_img.sum() == 0:
                    continue

                # Representative Undersampling
                if is_train:
                    if tile_mask.sum() > 0:
                        # Keep all positive tiles
                        all_tiles.append(tile_img)
                        all_masks.append(tile_mask)
                    else:
                        # Keep 20% of background tiles
                        if np.random.rand() < Config.BACKGROUND_SAMPLE_RATE:
                            all_tiles.append(tile_img)
                            all_masks.append(tile_mask)
                else:
                    # Keep all validation tiles for accurate metric calculation
                    all_tiles.append(tile_img)
                    all_masks.append(tile_mask)

    all_tiles_arr = np.array(all_tiles, dtype=np.uint8)
    all_masks_arr = np.array(all_masks, dtype=np.uint8)

    # Save to cache
    np.save(cache_path_tiles, all_tiles_arr)
    np.save(cache_path_masks, all_masks_arr)

    return all_tiles_arr, all_masks_arr


def prepare_train_val_loaders(load_cached_data=True):
    """
    Prepares DataLoaders for training and validation.
    Handles caching of processed tiles.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    train_tiles_path = Config.TRAIN_CACHE_PATH
    train_masks_path = train_tiles_path.replace("tiles.npy", "masks.npy")

    val_tiles_path = Config.VAL_CACHE_PATH
    val_masks_path = val_tiles_path.replace("tiles.npy", "masks.npy")

    # --- Load Training Data ---
    if (
        load_cached_data
        and os.path.exists(train_tiles_path)
        and os.path.exists(train_masks_path)
    ):
        print("Loading cached training data...")
        train_tiles = np.load(train_tiles_path)
        train_masks = np.load(train_masks_path)
    else:
        print("Processing and caching training data...")
        train_tiles, train_masks = process_and_cache_data(
            Config.TRAIN_METADATA_PATH,
            train_tiles_path,
            train_masks_path,
            is_train=True,
        )

    # --- Load Validation Data ---
    if (
        load_cached_data
        and os.path.exists(val_tiles_path)
        and os.path.exists(val_masks_path)
    ):
        print("Loading cached validation data...")
        val_tiles = np.load(val_tiles_path)
        val_masks = np.load(val_masks_path)
    else:
        print("Processing and caching validation data...")
        val_tiles, val_masks = process_and_cache_data(
            Config.VAL_METADATA_PATH, val_tiles_path, val_masks_path, is_train=False
        )

    print(f"Training Tiles: {len(train_tiles)}")
    print(f"Validation Tiles: {len(val_tiles)}")

    # --- Augmentations ---
    # Geometric Augmentations (Applied to 4-channel image + mask)
    train_geo_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5
            ),
        ]
    )

    # Color Augmentations (Applied to RGB channels only)
    train_color_transform = A.Compose(
        [
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
        ]
    )

    # --- Datasets & Loaders ---
    train_dataset = HuBMAPDataset(
        train_tiles,
        train_masks,
        transform=train_geo_transform,
        color_transform=train_color_transform,
    )

    val_dataset = HuBMAPDataset(
        val_tiles, val_masks, transform=None, color_transform=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader():
    """
    Returns a DataLoader for the test set.
    """
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    test_dataset = TestDataset(test_df)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
