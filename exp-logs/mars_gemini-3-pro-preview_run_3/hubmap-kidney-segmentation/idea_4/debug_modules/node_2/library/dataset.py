import os
import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import get_tissue_mask_from_json, rle_decode


class HuBMAPDataset(Dataset):
    """
    Dataset class for HuBMAP kidney tissue segmentation.
    Handles large TIFF loading, tile extraction, and mask management with caching.
    """

    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Whether to load generated tile lists/masks from cache.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.tile_size = Config.TILE_SIZE

        # Determine stride based on mode
        # Train: No overlap for efficiency
        # Val/Test: 50% overlap for stitching
        if self.mode == "train":
            self.stride = self.tile_size
        else:
            self.stride = self.tile_size // 2

        # Load metadata
        if self.mode == "train":
            self.metadata = pd.read_csv(Config.TRAIN_METADATA_PATH)
        elif self.mode == "validation":
            self.metadata = pd.read_csv(Config.VAL_METADATA_PATH)
        elif self.mode == "test":
            self.metadata = pd.read_csv(Config.TEST_METADATA_PATH)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Setup transforms
        self.transform = self._get_transforms()

        # Generate or load tile index
        self.tiles = self._generate_tile_index()

        # Handle Debug mode
        if Config.DEBUG:
            print(
                f"DEBUG MODE: Subsampling dataset from {len(self.tiles)} to {Config.DEBUG_SAMPLE_SIZE}"
            )
            self.tiles = self.tiles[: Config.DEBUG_SAMPLE_SIZE]

    def _get_transforms(self):
        """Returns albumentations transforms based on mode."""
        if self.mode == "train":
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    # Color augmentations only on RGB channels (0-255 range)
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def _generate_tile_index(self):
        """
        Generates a list of valid tiles.
        Caches the result to a parquet file to avoid re-computation.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(
            Config.CACHE_DIR, f"tiles_{self.mode}_{self.tile_size}.parquet"
        )

        if self.load_cached_data and os.path.exists(cache_path):
            try:
                df_tiles = pd.read_parquet(cache_path)
                return df_tiles.to_dict("records")
            except Exception as e:
                print(f"Failed to load cached tiles: {e}. Regenerating...")

        print(f"Generating tile index for {self.mode}...")
        tiles = []

        for _, row in self.metadata.iterrows():
            image_id = row["id"]

            # Construct full paths
            # Metadata paths are relative to input root (e.g., 'train/file.tiff')
            # But Config.INPUT_ROOT is './input'.
            # We need to ensure we join them correctly.
            # If metadata path starts with 'input/', strip it or handle it.
            # Based on previous context, metadata paths are like 'train/id.tiff'
            image_path = os.path.join(Config.INPUT_ROOT, row["image_path"])

            # Handle anatomical json path
            if "anatomical_json_path" in row and pd.notna(row["anatomical_json_path"]):
                anat_path = row["anatomical_json_path"]
            else:
                anat_path = None

            # Get image dimensions
            # We open the image to get exact dims if not in metadata,
            # but metadata usually has width_pixels/height_pixels.
            w = row["width_pixels"]
            h = row["height_pixels"]

            # Load anatomical mask for filtering (only for train/val usually)
            # For test, we also use it to restrict inference area if available.
            tissue_mask = None
            if anat_path:
                tissue_mask = get_tissue_mask_from_json(
                    image_id, anat_path, (h, w), load_cached_data=self.load_cached_data
                )

            # Generate grid
            # x, y are top-left coordinates
            x_points = range(0, w, self.stride)
            y_points = range(0, h, self.stride)

            for y in y_points:
                for x in x_points:
                    # Check boundary
                    # If tile goes off edge, shift it back to fit
                    x_eff = min(x, w - self.tile_size)
                    y_eff = min(y, h - self.tile_size)

                    # Ensure we don't have duplicates due to shifting at the end
                    if x_eff < 0 or y_eff < 0:
                        continue  # Image smaller than tile size? Unlikely but safe.

                    # Filter by tissue mask
                    if tissue_mask is not None:
                        # Extract mask patch
                        mask_patch = tissue_mask[
                            y_eff : y_eff + self.tile_size,
                            x_eff : x_eff + self.tile_size,
                        ]
                        # If intersection is too low (e.g., < 1%), skip
                        # For test set, we might want to be more lenient or keep all if unsure.
                        # Strategy: For Train, strict filter. For Val/Test, keep if any tissue.
                        if np.sum(mask_patch) == 0:
                            continue

                    tiles.append(
                        {
                            "id": image_id,
                            "image_path": row[
                                "image_path"
                            ],  # Keep relative for portability
                            "x": x_eff,
                            "y": y_eff,
                            "w": w,
                            "h": h,
                        }
                    )

            # Explicitly clear memory
            del tissue_mask

        # Save to cache
        df_tiles = pd.DataFrame(tiles)
        # Drop duplicates if any (due to edge shifting)
        df_tiles = df_tiles.drop_duplicates(subset=["id", "x", "y"])
        df_tiles.to_parquet(cache_path, index=False)

        return df_tiles.to_dict("records")

    def _get_mask_tile(self, image_id, x, y, h, w):
        """
        Retrieves the binary mask for a specific tile.
        Uses caching to store the full decoded mask as .npy to avoid repeated RLE decoding.
        """
        if self.mode == "test":
            return np.zeros((self.tile_size, self.tile_size), dtype=np.float32)

        # Cache path for the full mask
        mask_cache_path = os.path.join(Config.CACHE_DIR, f"{image_id}_mask.npy")

        # 1. Check if full mask .npy exists
        if self.load_cached_data and os.path.exists(mask_cache_path):
            # Load using mmap to avoid reading whole file into RAM
            full_mask = np.load(mask_cache_path, mmap_mode="r")
        else:
            # 2. Decode RLE and save
            # Find RLE in metadata
            row = self.metadata[self.metadata["id"] == image_id].iloc[0]
            if "encoding" in row and pd.notna(row["encoding"]):
                rle = row["encoding"]
                full_mask = rle_decode(rle, (h, w))
            else:
                # Fallback: try to load from JSON if RLE not present (rare for train.csv)
                # Or return zeros
                full_mask = np.zeros((h, w), dtype=np.uint8)

            # Save to cache
            np.save(mask_cache_path, full_mask)
            # Reload in mmap mode
            full_mask = np.load(mask_cache_path, mmap_mode="r")

        # Extract tile
        mask_tile = full_mask[y : y + self.tile_size, x : x + self.tile_size]
        return mask_tile.astype(np.float32)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile_info = self.tiles[idx]
        image_id = tile_info["id"]
        x, y = tile_info["x"], tile_info["y"]

        # Open image using rasterio
        full_image_path = os.path.join(Config.INPUT_ROOT, tile_info["image_path"])

        # Read image window
        # Note: rasterio window is ((row_start, row_stop), (col_start, col_stop))
        window = rasterio.windows.Window(x, y, self.tile_size, self.tile_size)

        with rasterio.open(full_image_path) as src:
            # Read RGB (channels 1, 2, 3)
            # rasterio reads as (C, H, W), we transpose to (H, W, C) for albumentations
            img = src.read([1, 2, 3], window=window)
            img = np.moveaxis(img, 0, -1)

        # Get mask
        mask = self._get_mask_tile(image_id, x, y, tile_info["h"], tile_info["w"])

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Ensure mask has channel dimension (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, mask

    def get_tile_metadata(self, idx):
        """
        Returns metadata for a specific tile index.
        Useful for stitching predictions during inference.
        """
        return self.tiles[idx]
