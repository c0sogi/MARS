import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import provided utility
from library.utils import create_tissue_mask


class HubmapDataset(Dataset):
    def __init__(
        self,
        metadata_df,
        root_dir="./input",
        transform=None,
        tile_size=512,
        stride=256,
        tissue_threshold=0.05,
        split="train",
        load_cached_data=True,
        cache_dir="./working/idea_5/",
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata containing image IDs and paths.
            root_dir (str): Root directory for input data.
            transform (albumentations.Compose): Augmentation pipeline.
            tile_size (int): Size of the tiles (H, W).
            stride (int): Stride for sliding window.
            tissue_threshold (float): Minimum fraction of tissue required to keep a tile.
            split (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Whether to use cached tile lists/masks.
            cache_dir (str): Directory to store cached files.
        """
        self.metadata_df = metadata_df
        self.root_dir = root_dir
        self.transform = transform
        self.tile_size = tile_size
        self.stride = stride
        self.tissue_threshold = tissue_threshold
        self.split = split
        self.load_cached_data = load_cached_data
        self.cache_dir = cache_dir

        # Ensure cache directories exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "gt_masks"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "tissue_masks_cache"), exist_ok=True)

        # Prepare tiles
        self.tiles = self._prepare_tiles()

    def update_resolution(self, new_tile_size, new_stride=None):
        """
        Updates the resolution for Progressive Resizing.
        """
        self.tile_size = new_tile_size
        if new_stride is not None:
            self.stride = new_stride
        else:
            # Default: scale stride roughly with tile size (e.g., 50% overlap)
            self.stride = new_tile_size // 2

        print(
            f"Updating dataset resolution to {self.tile_size}x{self.tile_size} with stride {self.stride}..."
        )
        self.tiles = self._prepare_tiles()

    def _rle_decode(self, rle_string, shape):
        """
        Decodes RLE string to binary mask.
        shape: (height, width)
        """
        if pd.isna(rle_string):
            return np.zeros(shape, dtype=np.uint8)

        s = rle_string.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths

        # Create flat array
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1

        # Reshape (Fortran-style: top-down then left-right)
        return img.reshape(shape, order="F")

    def _get_ground_truth_mask(self, image_id, rle, shape):
        """
        Retrieves the full ground truth mask. Uses caching with memory mapping.
        """
        cache_path = os.path.join(
            self.cache_dir, "gt_masks", f"{image_id}_{shape[0]}x{shape[1]}_mask.npy"
        )

        if self.load_cached_data and os.path.exists(cache_path):
            try:
                # Use mmap_mode='r' to avoid loading full mask into RAM
                mask = np.load(cache_path, mmap_mode="r")
                if mask.shape == shape:
                    return mask
            except Exception:
                pass

        # Compute if not cached
        mask = self._rle_decode(rle, shape)
        np.save(cache_path, mask)

        # Return mmap to save memory
        return np.load(cache_path, mmap_mode="r")

    def _prepare_tiles(self):
        """
        Generates a list of valid tiles based on tissue masks.
        Returns a list of dicts: [{'image_id': ..., 'x': ..., 'y': ...}, ...]
        """
        cache_name = f"tiles_{self.split}_{self.tile_size}_{self.stride}_{self.tissue_threshold}.parquet"
        cache_path = os.path.join(self.cache_dir, cache_name)

        if self.load_cached_data and os.path.exists(cache_path):
            return pd.read_parquet(cache_path).to_dict("records")

        tiles = []

        # For test set, we might not want to filter by tissue threshold strictly if we want full coverage,
        # but the prompt says "Detect FTUs...". Usually we still filter background.
        # We will apply the same logic.

        for _, row in self.metadata_df.iterrows():
            image_id = row["id"]
            img_path = os.path.join(self.root_dir, row["image_path"])
            anat_path = os.path.join(self.root_dir, row["anatomical_json_path"])

            # Determine dimensions
            if (
                "width_pixels" in row
                and "height_pixels" in row
                and not pd.isna(row["width_pixels"])
            ):
                W, H = int(row["width_pixels"]), int(row["height_pixels"])
            else:
                with rasterio.open(img_path) as src:
                    H, W = src.height, src.width

            # Load tissue mask (Cached via library function)
            tissue_mask = create_tissue_mask(
                anat_path,
                H,
                W,
                load_cached_data=self.load_cached_data,
                cache_dir=os.path.join(self.cache_dir, "tissue_masks_cache"),
            )

            # Generate coordinates
            x_points = range(0, W - self.tile_size + 1, self.stride)
            y_points = range(0, H - self.tile_size + 1, self.stride)

            # Handle small image edge case
            if W < self.tile_size:
                x_points = [0]
            if H < self.tile_size:
                y_points = [0]

            for y in y_points:
                for x in x_points:
                    # Check tissue overlap
                    mask_crop = tissue_mask[
                        y : y + self.tile_size, x : x + self.tile_size
                    ]

                    # If enough tissue is present, keep the tile
                    if mask_crop.mean() > self.tissue_threshold:
                        tiles.append(
                            {"image_id": image_id, "x": x, "y": y, "h": H, "w": W}
                        )

        # Save to cache
        df = pd.DataFrame(tiles)
        if not df.empty:
            df.to_parquet(cache_path)
        else:
            # Fallback if no tiles found (e.g. threshold too high or bad masks)
            # Create at least one dummy tile or handle upstream.
            # For now, we save empty parquet.
            pd.DataFrame(columns=["image_id", "x", "y", "h", "w"]).to_parquet(
                cache_path
            )

        return tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile_info = self.tiles[idx]
        image_id = tile_info["image_id"]
        x, y = tile_info["x"], tile_info["y"]

        # Get metadata row
        row = self.metadata_df[self.metadata_df["id"] == image_id].iloc[0]
        img_path = os.path.join(self.root_dir, row["image_path"])

        # 1. Read Image Tile
        window = Window(x, y, self.tile_size, self.tile_size)

        with rasterio.open(img_path) as src:
            # Read RGB
            img = src.read(window=window)
            img = np.moveaxis(img, 0, -1)  # (H, W, C)

            # Ensure 3 channels
            if img.shape[2] > 3:
                img = img[:, :, :3]

            # Pad if tile is smaller than expected (e.g. at edges if logic forced it)
            if img.shape[0] != self.tile_size or img.shape[1] != self.tile_size:
                pad_h = self.tile_size - img.shape[0]
                pad_w = self.tile_size - img.shape[1]
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

        # 2. Get Mask Tile
        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)

        if self.split in ["train", "validation"]:
            rle = row["encoding"]
            H, W = tile_info["h"], tile_info["w"]

            # Get full mask
            full_mask = self._get_ground_truth_mask(image_id, rle, (H, W))

            # Crop mask
            mask_crop = full_mask[y : y + self.tile_size, x : x + self.tile_size]

            # Pad mask if needed
            if (
                mask_crop.shape[0] != self.tile_size
                or mask_crop.shape[1] != self.tile_size
            ):
                pad_h = self.tile_size - mask_crop.shape[0]
                pad_w = self.tile_size - mask_crop.shape[1]
                mask_crop = np.pad(mask_crop, ((0, pad_h), (0, pad_w)), mode="constant")

            mask = mask_crop.astype(np.float32)

        # 3. Augmentations
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            # Default transform
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        # Ensure mask has channel dimension
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, mask
