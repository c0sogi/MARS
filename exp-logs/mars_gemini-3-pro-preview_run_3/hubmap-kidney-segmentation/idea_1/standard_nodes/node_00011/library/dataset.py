import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import get_tissue_mask, polygons_to_mask


class HuBMAPDataset(Dataset):
    """
    PyTorch Dataset for HuBMAP Kidney Segmentation.
    Implements tissue-aware tiling, caching, and on-the-fly mask generation.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        mode: str = "train",
        transform: A.Compose = None,
        load_cached_data: bool = True,
        tissue_overlap_threshold: float = 0.05,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Dataframe containing image metadata (paths, ids).
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            load_cached_data (bool): Whether to use cached tile metadata.
            tissue_overlap_threshold (float): Minimum fraction of tissue required to keep a tile.
        """
        self.metadata_df = metadata_df
        self.mode = mode
        self.transform = transform
        self.tissue_overlap_threshold = tissue_overlap_threshold

        # Define cache path for tile metadata
        self.cache_dir = os.path.join(Config.WORKING_DIR, "idea_1")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(
            self.cache_dir, f"tiles_{mode}_{Config.TILE_SIZE}_{Config.STRIDE}.parquet"
        )

        # 1. Generate or Load Tile List
        self.tiles = self._prepare_tiles(load_cached_data)

        # 2. Pre-load Glomerulus Annotations (for Train/Val only)
        # We store polygons in memory to avoid reading JSONs in __getitem__
        self.annotations = {}
        if self.mode in ["train", "validation"]:
            self._preload_annotations()

        # Define default normalization if no transform provided
        if self.transform is None:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def _prepare_tiles(self, load_cached_data: bool) -> pd.DataFrame:
        """
        Generates a list of valid tiles that intersect with tissue regions.
        Uses caching to avoid re-computation.
        """
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                # print(f"Loading cached tiles from {self.cache_path}")
                return pd.read_parquet(self.cache_path)
            except Exception:
                pass  # Fallback to re-computation

        # print(f"Generating tiles for {self.mode} set...")
        tile_data = []

        for _, row in self.metadata_df.iterrows():
            image_id = row["id"]
            # Construct full paths
            image_path = os.path.join(Config.INPUT_DIR, row["image_path"])
            anat_path = os.path.join(Config.INPUT_DIR, row["anatomical_json_path"])

            # Open image to get dimensions
            with rasterio.open(image_path) as src:
                H, W = src.height, src.width

            # Get Tissue Mask (Cortex/Medulla)
            # This uses the utility function which handles its own caching
            tissue_mask = get_tissue_mask(
                anat_path,
                (H, W),
                valid_classes=["Cortex", "Medulla"],
                load_cached_data=load_cached_data,
            )

            # Generate Sliding Window Coordinates
            # x = col_off, y = row_off
            for y in range(0, H, Config.STRIDE):
                for x in range(0, W, Config.STRIDE):
                    # Adjust for edges (though we will use boundless read later,
                    # we want to define the 'core' position)

                    # Check intersection with tissue mask
                    # We look at the region in the tissue mask corresponding to this tile
                    # Handle edge clipping for the mask check
                    y_end = min(y + Config.TILE_SIZE, H)
                    x_end = min(x + Config.TILE_SIZE, W)

                    mask_crop = tissue_mask[y:y_end, x:x_end]

                    if mask_crop.size > 0:
                        tissue_fraction = np.sum(mask_crop) / (
                            Config.TILE_SIZE * Config.TILE_SIZE
                        )
                    else:
                        tissue_fraction = 0

                    # Keep tile if it has enough tissue or if we are in test mode (be safer in test)
                    # For test, we might want to be more generous, but usually Cortex is sufficient.
                    threshold = (
                        self.tissue_overlap_threshold if self.mode != "test" else 0.001
                    )

                    if tissue_fraction >= threshold:
                        tile_data.append(
                            {
                                "id": image_id,
                                "image_path": row["image_path"],
                                "json_path": row.get(
                                    "json_path", None
                                ),  # Might be NaN/None for test
                                "x": x,
                                "y": y,
                                "h": H,
                                "w": W,
                            }
                        )

        df_tiles = pd.DataFrame(tile_data)
        # Save to cache
        df_tiles.to_parquet(self.cache_path, index=False)
        return df_tiles

    def _preload_annotations(self):
        """
        Loads all glomerulus polygons into memory for faster access.
        """
        unique_ids = self.metadata_df["id"].unique()

        for image_id in unique_ids:
            # Find the row
            row = self.metadata_df[self.metadata_df["id"] == image_id].iloc[0]
            json_rel_path = row.get("json_path")

            if pd.isna(json_rel_path):
                self.annotations[image_id] = []
                continue

            json_path = os.path.join(Config.INPUT_DIR, json_rel_path)

            polygons = []
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)

                    # Parse standard GeoJSON-like structure from description
                    for feature in data:
                        geom = feature.get("geometry", {})
                        coords = geom.get("coordinates", [])
                        # Coords are typically list of lists of [x, y]
                        # Flatten if necessary or append directly
                        polygons.extend(coords)
                except Exception:
                    pass  # Handle empty or corrupt files gracefully

            self.annotations[image_id] = polygons

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        row = self.tiles.iloc[idx]
        image_id = row["id"]
        x, y = int(row["x"]), int(row["y"])

        # 1. Load Image Tile
        full_img_path = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Use rasterio to read specific window
        # boundless=True automatically pads with 0 if window extends beyond image
        window = Window(
            col_off=x, row_off=y, width=Config.TILE_SIZE, height=Config.TILE_SIZE
        )

        with rasterio.open(full_img_path) as src:
            # Read image: shape (C, H, W)
            img = src.read(window=window, boundless=True, fill_value=0)

            # Handle channels
            if src.count == 1:
                # Duplicate to 3 channels
                img = np.repeat(img, 3, axis=0)
            elif src.count > 3:
                # Take first 3 channels if more exist
                img = img[:3, :, :]

        # Convert to HWC for Albumentations/OpenCV
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        img = img.astype(np.uint8)

        # 2. Generate Mask (Train/Val only)
        mask = np.zeros((Config.TILE_SIZE, Config.TILE_SIZE), dtype=np.float32)

        if self.mode in ["train", "validation"]:
            polygons = self.annotations.get(image_id, [])

            if polygons:
                # Filter and shift polygons
                # We simply shift all polygons by (-x, -y) and let cv2.fillPoly handle clipping
                # This is computationally cheaper than geometric intersection for thousands of polys
                # provided we don't have millions of points.

                shifted_polys = []
                for poly in polygons:
                    # poly is a list of [px, py]
                    p_arr = np.array(poly, dtype=np.int32)
                    p_arr -= np.array([x, y], dtype=np.int32)
                    shifted_polys.append(p_arr)

                # Draw on mask
                # 1.0 for glomerulus, 0.0 for background
                cv2.fillPoly(mask, shifted_polys, 1.0)

        # 3. Augmentations
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented["image"]
            mask_tensor = augmented["mask"]
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask).float()

        # Ensure mask has channel dimension (1, H, W)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        # 4. Return
        # For training, we need (image, mask)
        # For inference/debugging, metadata is useful.
        # PyTorch DataLoader expects tensors. We can return a dict or tuple.
        # We will return a dictionary which is flexible.

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "id": image_id,
            "x": torch.tensor(x, dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.long),
        }
