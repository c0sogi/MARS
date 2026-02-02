import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
import cv2

# Import from provided libraries
from library.config import Config
from library.utils import rle_decode, get_tissue_mask


class HubmapDataset(Dataset):
    """
    PyTorch Dataset for HuBMAP FTU Detection.

    Implements Hierarchical ROI Sampling:
    - Training: Dynamically samples tiles centered on valid tissue regions (Cortex/Medulla).
      Caches ground truth masks and valid coordinate indices to disk for efficiency.
    - Validation/Test: Returns image metadata and paths to allow for sliding window inference
      in the evaluation loop.
    """

    def __init__(
        self,
        mode="train",
        transform=None,
        samples_per_epoch=None,
        cache_dir=Config.CACHE_DIR,
    ):
        """
        Args:
            mode (str): One of 'train', 'val', 'test'.
            transform (callable, optional): Albumentations transform pipeline.
            samples_per_epoch (int, optional): Total number of samples per epoch for training.
                                               If None, defaults to len(images) * 200.
            cache_dir (str): Directory to store cached .npy files.
        """
        self.mode = mode
        self.transform = transform
        self.cfg = Config
        self.cache_dir = cache_dir

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load Metadata based on mode
        if self.mode == "train":
            self.df = pd.read_csv(self.cfg.TRAIN_METADATA_PATH)
            # Define dataset length
            if samples_per_epoch:
                self.length = samples_per_epoch
            else:
                # Default: 200 samples per image per epoch
                self.length = len(self.df) * 200

            # Pre-compute/Cache data for training efficiency
            self._prepare_training_cache()

        elif self.mode == "val":
            self.df = pd.read_csv(self.cfg.VAL_METADATA_PATH)
            # Sample multiple crops per image for validation coverage
            self.length = len(self.df) * 20
            self._prepare_training_cache()

        elif self.mode == "test":
            self.df = pd.read_csv(self.cfg.TEST_METADATA_PATH)
            self.length = len(self.df)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _prepare_training_cache(self):
        """
        Generates and caches:
        1. Full binary Ground Truth masks (from RLE) -> .npy
        2. Valid Tissue Coordinates (downsampled) -> .npy

        This ensures __getitem__ is fast and deterministic.
        """
        print(f"Checking/Preparing cache for {len(self.df)} training images...")

        for _, row in self.df.iterrows():
            img_id = row["id"]
            h = row["height_pixels"]
            w = row["width_pixels"]

            # 1. Cache Ground Truth Binary Mask
            gt_path = os.path.join(self.cache_dir, f"{img_id}_gt_mask.npy")
            if not os.path.exists(gt_path):
                # Decode RLE to binary mask
                if "encoding" in row and pd.notna(row["encoding"]):
                    mask = rle_decode(row["encoding"], (h, w))
                else:
                    mask = np.zeros((h, w), dtype=np.uint8)
                np.save(gt_path, mask)
                # Explicitly delete to free memory
                del mask

            # 2. Cache Valid Tissue Indices (Downsampled for memory efficiency)
            indices_path = os.path.join(self.cache_dir, f"{img_id}_valid_indices.npy")
            if not os.path.exists(indices_path):
                # Load tissue mask using utility (handles its own caching)
                # anatomical_json_path is relative, e.g., "train/id.json"
                # get_tissue_mask expects path relative to input or absolute
                tissue_mask = get_tissue_mask(
                    img_id, w, h, row["anatomical_json_path"], load_cached_data=True
                )

                # Downsample mask to find valid coordinates efficiently
                # Scale factor 32 reduces 50k x 50k -> ~1.5k x 1.5k
                scale = 32
                small_h, small_w = h // scale, w // scale

                # Resize using Nearest Neighbor to preserve binary nature
                if small_h > 0 and small_w > 0:
                    small_mask = cv2.resize(
                        tissue_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST
                    )
                else:
                    small_mask = np.zeros((1, 1), dtype=np.uint8)

                # Get coordinates of tissue pixels
                ys, xs = np.where(small_mask > 0)

                if len(ys) > 0:
                    # Scale back to original resolution
                    # Add offset to point to center of the 32x32 block
                    indices = np.stack([ys, xs], axis=1).astype(np.int32) * scale
                    indices += scale // 2
                else:
                    # Fallback: if no tissue, use center of image
                    indices = np.array([[h // 2, w // 2]], dtype=np.int32)

                np.save(indices_path, indices)
                del tissue_mask, small_mask, indices

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode in ["train", "val"]:
            return self._get_train_item(idx)
        else:
            return self._get_inference_item(idx)

    def _get_train_item(self, idx):
        """
        Fetches a random tile centered on valid tissue for training.
        """
        # Map linear index to image
        img_idx = idx % len(self.df)
        row = self.df.iloc[img_idx]
        img_id = row["id"]
        h, w = row["height_pixels"], row["width_pixels"]

        # Load valid sampling coordinates
        indices_path = os.path.join(self.cache_dir, f"{img_id}_valid_indices.npy")
        valid_indices = np.load(indices_path, mmap_mode="r")

        # Sample a center point
        if len(valid_indices) > 0:
            rand_i = np.random.randint(0, len(valid_indices))
            cy, cx = valid_indices[rand_i]

            # Add spatial jitter (random shift)
            if self.mode == "train":
                jitter = 64
                cy += np.random.randint(-jitter, jitter)
                cx += np.random.randint(-jitter, jitter)
        else:
            cy, cx = h // 2, w // 2

        # Define crop window
        tile_size = self.cfg.TILE_SIZE

        # Clamp center to stay within image bounds
        cy = np.clip(cy, tile_size // 2, h - tile_size // 2)
        cx = np.clip(cx, tile_size // 2, w - tile_size // 2)

        y_min = int(cy - tile_size // 2)
        x_min = int(cx - tile_size // 2)

        # 1. Read Image Tile (RGB)
        img_path = os.path.join(self.cfg.INPUT_DIR, row["image_path"])
        with rasterio.open(img_path) as src:
            window = Window(x_min, y_min, tile_size, tile_size)
            # Check channel count (Cite debug_lesson_6)
            if src.count >= 3:
                img = src.read([1, 2, 3], window=window)
            else:
                img = src.read(1, window=window)
                img = np.stack([img, img, img], axis=0)

            # Transpose to (H, W, C) for Albumentations/OpenCV
            img = np.moveaxis(img, 0, -1)

        # 2. Read Mask Tile
        gt_path = os.path.join(self.cache_dir, f"{img_id}_gt_mask.npy")
        # Use mmap_mode='r' to read only the specific window from disk
        full_mask = np.load(gt_path, mmap_mode="r")
        mask = full_mask[y_min : y_min + tile_size, x_min : x_min + tile_size].copy()

        # 3. Augmentations
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # 4. Convert to Tensor
        # If transform returns numpy, convert to torch
        if not isinstance(img, torch.Tensor):
            # (H, W, C) -> (C, H, W)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()

        # Ensure mask has channel dimension if needed (1, H, W) or just (H, W)
        # Usually Loss functions expect (B, H, W) or (B, 1, H, W).
        # Let's return (1, H, W) for mask to match model output channel.
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, mask

    def _get_inference_item(self, idx):
        """
        Returns metadata for Validation/Test.
        The inference loop handles sliding window tiling to ensure full coverage.
        """
        row = self.df.iloc[idx]

        # Construct full path
        img_path = os.path.join(self.cfg.INPUT_DIR, row["image_path"])

        # Return dict with all necessary info
        sample = {
            "id": row["id"],
            "image_path": img_path,
            "anatomical_json_path": row["anatomical_json_path"],
            "height": row["height_pixels"],
            "width": row["width_pixels"],
        }

        # Include RLE for validation scoring if available
        if "encoding" in row and pd.notna(row["encoding"]):
            sample["rle"] = row["encoding"]
        else:
            sample["rle"] = ""

        return sample
