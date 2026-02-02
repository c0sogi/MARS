import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode


class HuBMAPDataset(Dataset):
    """
    PyTorch Dataset for HuBMAP FTU Detection.
    Handles tiling of large TIFF images, mask decoding, caching, and augmentation.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        split: str = "train",
        transform=None,
        load_cached_data: bool = True,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata containing image IDs and paths.
            split (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
            load_cached_data (bool): Whether to load cached tile lists and masks.
        """
        self.df = metadata_df
        self.split = split
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Setup directories
        self.mask_cache_dir = os.path.join(Config.WORKING_DIR, "masks")
        os.makedirs(self.mask_cache_dir, exist_ok=True)

        self.tile_cache_path = os.path.join(Config.WORKING_DIR, f"{split}_tiles.npy")

        # Define default transforms if none provided
        if self.transform is None:
            if self.split == "train":
                self.transform = A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=0.5,
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=0.5
                        ),
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose(
                    [
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                        ToTensorV2(),
                    ]
                )

        # Prepare tile list
        self.tiles = self._prepare_tiles()

    def _prepare_tiles(self):
        """
        Generates a list of tiles (coordinates) for the dataset.
        Implements caching and balanced sampling.
        """
        # Try loading cache
        if self.load_cached_data and os.path.exists(self.tile_cache_path):
            try:
                tiles = np.load(self.tile_cache_path, allow_pickle=True).tolist()
                return tiles
            except Exception:
                pass  # Fallback to re-generation if cache is corrupt

        tiles = []

        # Determine stride based on split
        if self.split == "test":
            # Overlapping tiles for inference
            stride = int(Config.TILE_SIZE * (1 - Config.INFERENCE_OVERLAP))
        else:
            # Non-overlapping for training/validation
            stride = Config.TILE_SIZE

        # print(f"Generating tiles for {self.split} set...")

        for _, row in self.df.iterrows():
            img_id = row["id"]
            # Ensure dimensions are integers
            img_h = int(row["height_pixels"])
            img_w = int(row["width_pixels"])

            # 1. Handle Mask Caching (for Train/Val)
            mask_path = os.path.join(self.mask_cache_dir, f"{img_id}.npy")
            has_mask = "encoding" in row and pd.notna(row["encoding"])

            if (self.split in ["train", "val"]) and has_mask:
                if (not self.load_cached_data) or (not os.path.exists(mask_path)):
                    # Decode RLE and save as binary mask
                    mask = rle_decode(row["encoding"], (img_h, img_w))
                    np.save(mask_path, mask)

            # 2. Generate Grid
            # Iterate top-left corners
            y_points = range(0, img_h, stride)
            x_points = range(0, img_w, stride)

            img_tiles = []

            # Open mask in memory-mapped mode for efficient checking
            mask_acc = None
            if (
                (self.split in ["train", "val"])
                and has_mask
                and os.path.exists(mask_path)
            ):
                mask_acc = np.load(mask_path, mmap_mode="r")

            for y in y_points:
                for x in x_points:
                    # Determine if tile contains foreground
                    is_fg = False
                    if mask_acc is not None:
                        # Calculate read dimensions (handling edges)
                        h_read = min(Config.TILE_SIZE, img_h - y)
                        w_read = min(Config.TILE_SIZE, img_w - x)

                        # Check slice
                        # We only need to know if ANY pixel is 1
                        if np.any(mask_acc[y : y + h_read, x : x + w_read]):
                            is_fg = True

                    img_tiles.append(
                        {
                            "id": img_id,
                            "image_path": row["image_path"],
                            "x": x,
                            "y": y,
                            "h": img_h,
                            "w": img_w,
                            "is_fg": is_fg,
                        }
                    )

            # 3. Apply Sampling Strategy (Only for Train)
            if self.split == "train":
                fg_tiles = [t for t in img_tiles if t["is_fg"]]
                bg_tiles = [t for t in img_tiles if not t["is_fg"]]

                # Undersample background to match foreground (Cite solution_lesson_node_00004)
                n_fg = len(fg_tiles)
                n_bg = int(n_fg * Config.BACKGROUND_RATIO)

                # Ensure we don't sample more than available
                n_bg = min(n_bg, len(bg_tiles))

                if n_bg > 0:
                    # Use numpy choice for reproducibility
                    bg_indices = np.random.choice(len(bg_tiles), n_bg, replace=False)
                    keep_bg = [bg_tiles[i] for i in bg_indices]
                else:
                    keep_bg = []

                tiles.extend(fg_tiles + keep_bg)
            else:
                # Keep all tiles for val/test
                tiles.extend(img_tiles)

        # Save result to cache
        np.save(self.tile_cache_path, tiles)
        return tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile_info = self.tiles[idx]
        img_id = tile_info["id"]
        x, y = tile_info["x"], tile_info["y"]
        img_h, img_w = tile_info["h"], tile_info["w"]

        # Construct full path
        img_path = os.path.join(Config.INPUT_DIR, tile_info["image_path"])

        # --- Read Image ---
        # Calculate window size
        w_read = min(Config.TILE_SIZE, img_w - x)
        h_read = min(Config.TILE_SIZE, img_h - y)

        with rasterio.open(img_path) as src:
            # Read RGB channels (1, 2, 3)
            # Window(col_off, row_off, width, height)
            window = Window(x, y, w_read, h_read)
            if src.count >= 3:
                img = src.read([1, 2, 3], window=window)
            else:
                img = src.read([1], window=window)
                img = np.repeat(img, 3, axis=0)

        # Pad image if edge tile
        # img shape is (C, H, W)
        if h_read < Config.TILE_SIZE or w_read < Config.TILE_SIZE:
            pad_h = Config.TILE_SIZE - h_read
            pad_w = Config.TILE_SIZE - w_read
            # Pad: ((0,0), (0, pad_h), (0, pad_w)) -> Pad H and W, keep C
            img = np.pad(
                img,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

        # Transpose to (H, W, C) for Albumentations
        img = np.transpose(img, (1, 2, 0)).astype(np.uint8)

        # --- Read Mask ---
        mask = np.zeros((Config.TILE_SIZE, Config.TILE_SIZE), dtype=np.float32)

        if self.split in ["train", "val"]:
            mask_path = os.path.join(self.mask_cache_dir, f"{img_id}.npy")
            if os.path.exists(mask_path):
                # Use mmap to read specific crop
                mask_acc = np.load(mask_path, mmap_mode="r")
                mask_crop = mask_acc[y : y + h_read, x : x + w_read]

                # Copy to memory to allow padding/transforms
                mask_crop = np.array(mask_crop, dtype=np.float32)

                # Place crop into padded mask
                mask[:h_read, :w_read] = mask_crop

        # --- Augmentation ---
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Mask handling for PyTorch
        # Albumentations returns mask as (H, W) if input was (H, W).
        # We need (1, H, W) for the model output channel dimension.
        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif isinstance(mask, np.ndarray) and mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        return img, mask, tile_info
