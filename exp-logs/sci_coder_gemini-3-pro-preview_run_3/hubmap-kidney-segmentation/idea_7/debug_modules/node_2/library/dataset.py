import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode, get_tissue_mask


class HubmapDataset(Dataset):
    """
    Dataset class for HuBMAP FTU Detection.
    Handles large TIFF reading, RLE decoding with caching, ROI-based sampling,
    and Progressive Resizing.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        phase: str = "train",
        image_size: int = 512,
        samples_per_epoch: int = None,
        load_cached_data: bool = True,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Dataframe containing image IDs and metadata.
            phase (str): 'train', 'validation', or 'test'.
            image_size (int): Size of the tiles to extract (e.g., 512, 768).
            samples_per_epoch (int, optional): Number of samples to draw per image per epoch.
                                               If None, defaults to a heuristic based on image size.
            load_cached_data (bool): Whether to use cached mask files.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.phase = phase
        self.image_size = image_size
        self.load_cached_data = load_cached_data

        # Default samples per image if not provided
        if samples_per_epoch is None:
            self.samples_per_image = 100 if phase == "train" else 20
        else:
            self.samples_per_image = samples_per_epoch

        # Define Augmentations
        self.transform = self.get_transforms()

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def get_transforms(self):
        """
        Returns Albumentations composition based on the current phase.
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.phase == "train":
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=Config.AUG_PROB,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=Config.AUG_PROB,
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
        else:
            # Validation/Test: Normalize only
            return A.Compose(
                [
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        """
        Length is defined as number of images * samples_per_image.
        This creates a synthetic epoch length.
        """
        return len(self.df) * self.samples_per_image

    def _get_ground_truth_mask(self, image_id, rle, shape):
        """
        Retrieves the Ground Truth (FTU) mask.
        Implements caching: checks for .npy file, otherwise decodes RLE and saves.
        Uses mmap_mode='r' to keep memory usage low.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"{image_id}_gt_mask.npy")

        # 1. Try loading from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                # Load with mmap to avoid reading full array into RAM
                return np.load(cache_path, mmap_mode="r")
            except Exception:
                pass  # Fallback to regenerate

        # 2. Generate from RLE
        if pd.isna(rle) or rle == "":
            mask = np.zeros(shape, dtype=np.uint8)
        else:
            mask = rle_decode(rle, shape)

        # 3. Save to cache
        try:
            np.save(cache_path, mask)
            # Reload as mmap
            mask = np.load(cache_path, mmap_mode="r")
        except Exception:
            pass  # If save fails, return in-memory mask

        return mask

    def __getitem__(self, idx):
        # Map linear index to image index
        img_idx = idx // self.samples_per_image
        row = self.df.iloc[img_idx]

        image_id = row["id"]
        image_path = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Determine dimensions (handle potential missing metadata by opening file if needed)
        if "width_pixels" in row and "height_pixels" in row:
            W, H = int(row["width_pixels"]), int(row["height_pixels"])
        else:
            with rasterio.open(image_path) as src:
                H, W = src.height, src.width

        # 1. Load Tissue Mask (ROI)
        # We use the utility function which handles caching.
        # Note: utils.get_tissue_mask returns a numpy array.
        # To optimize, we check if the cache file exists and mmap it manually if possible,
        # otherwise we use the function as is.
        anatomical_json = row.get("anatomical_json_path", None)
        tissue_mask_cache_path = os.path.join(
            Config.CACHE_DIR, f"{image_id}_tissue_mask.npy"
        )

        if self.load_cached_data and os.path.exists(tissue_mask_cache_path):
            tissue_mask = np.load(tissue_mask_cache_path, mmap_mode="r")
        else:
            # Generate and allow utils to save it
            tissue_mask = get_tissue_mask(
                image_id, W, H, anatomical_json, self.load_cached_data
            )
            # If it was just saved, try to reload as mmap to save memory
            if os.path.exists(tissue_mask_cache_path):
                tissue_mask = np.load(tissue_mask_cache_path, mmap_mode="r")

        # 2. Load Ground Truth Mask (if training/val)
        gt_mask = None
        if self.phase in ["train", "validation"]:
            rle = row.get("encoding", None)
            gt_mask = self._get_ground_truth_mask(image_id, rle, (H, W))

        # 3. Sampling Logic (Rejection Sampling)
        # Try to find a tile with tissue content > threshold
        pad = self.image_size // 2  # Not used directly, but logic implies we need space

        x, y = 0, 0
        found_valid = False

        # Limit retries to avoid infinite loops on empty masks
        max_retries = 20 if self.phase == "train" else 5

        for _ in range(max_retries):
            # Random top-left coordinate
            x = np.random.randint(0, max(1, W - self.image_size))
            y = np.random.randint(0, max(1, H - self.image_size))

            # Crop tissue mask to check density
            # Handle edge case where image is smaller than tile size
            w_crop = min(self.image_size, W - x)
            h_crop = min(self.image_size, H - y)

            tissue_crop = tissue_mask[y : y + h_crop, x : x + w_crop]

            # Check tissue density
            if np.mean(tissue_crop) > Config.TISSUE_AREA_THRESHOLD:
                found_valid = True
                break

        # If Fail-Open ROI is active and we didn't find a valid spot,
        # we accept the last random coordinate (or 0,0)
        if not found_valid and not Config.FAIL_OPEN_ROI:
            # If strict, force 0,0 (or handle differently)
            x, y = 0, 0

        # 4. Read Image Tile using Rasterio Window
        # This is memory efficient
        window = Window(x, y, self.image_size, self.image_size)

        with rasterio.open(image_path) as src:
            # Handle cases where image < tile_size by padding automatically via rasterio or manual
            # Rasterio read with boundless=True pads automatically
            img = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
            # Move channels to last: (C, H, W) -> (H, W, C)
            img = np.transpose(img, (1, 2, 0))

        # 5. Extract Mask Tile
        mask_tile = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        if gt_mask is not None:
            # Slicing mmap array
            # Handle boundary conditions manually for numpy array
            h_real = min(self.image_size, H - y)
            w_real = min(self.image_size, W - x)

            if h_real > 0 and w_real > 0:
                mask_crop = gt_mask[y : y + h_real, x : x + w_real]
                mask_tile[:h_real, :w_real] = mask_crop

        # 6. Augmentation
        if self.transform:
            augmented = self.transform(image=img, mask=mask_tile)
            img = augmented["image"]
            mask_tile = augmented["mask"]

        # Ensure mask is channel-first for PyTorch: (H, W) -> (1, H, W)
        if isinstance(mask_tile, np.ndarray):
            if mask_tile.ndim == 2:
                mask_tile = torch.from_numpy(mask_tile).unsqueeze(0).float()
            else:
                mask_tile = torch.from_numpy(mask_tile).float()
        elif isinstance(mask_tile, torch.Tensor):
            if mask_tile.ndim == 2:
                mask_tile = mask_tile.unsqueeze(0).float()
            else:
                mask_tile = mask_tile.float()

        return img, mask_tile
