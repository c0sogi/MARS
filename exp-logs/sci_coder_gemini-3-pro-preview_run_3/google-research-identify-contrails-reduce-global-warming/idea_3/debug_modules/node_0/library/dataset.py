import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Identification.

    Loads satellite imagery bands, computes physics-informed 'Ash' false-color composites,
    calculates temporal differences, and constructs a 6-channel input tensor.
    """

    def __init__(self, split="train", load_cached_data=True, debug=False):
        """
        Args:
            split (str): One of 'train', 'validation', 'test'.
            load_cached_data (bool): Whether to load/save metadata from/to cache.
            debug (bool): If True, limits the dataset size for debugging.
        """
        self.split = split
        self.debug = debug
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache")

        # Load metadata
        self.metadata = self._load_metadata(split, load_cached_data)

        # Apply debug limit if configured
        limit = Config.get_dataset_limit()
        if self.debug and limit:
            self.metadata = self.metadata.iloc[:limit].reset_index(drop=True)

        # Define Augmentations
        if self.split == "train":
            self.transform = A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose([ToTensorV2()])

    def _load_metadata(self, split, load_cached_data):
        """
        Loads metadata with caching mechanism.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"{split}_metadata.parquet")

        # 1. Try to load cached data
        if load_cached_data and os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass  # Fallback to loading from source

        # 2. Load from source CSVs
        if split == "train":
            path = Config.TRAIN_METADATA_PATH
        elif split == "validation":
            path = Config.VAL_METADATA_PATH
        elif split == "test":
            path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        df = pd.read_csv(path)

        # 3. Save to cache
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass  # Non-critical failure

        return df

    def _normalize_range(self, data, vmin, vmax):
        """
        Normalizes data to [0, 1] based on physical bounds.
        """
        return (data - vmin) / (vmax - vmin)

    def _get_ash_color(self, b11, b13, b14, b15):
        """
        Computes the Ash false-color composite from brightness temperatures.

        Args:
            b11, b13, b14, b15: 2D arrays of brightness temperatures.

        Returns:
            np.ndarray: (H, W, 3) array with values in [0, 1].
        """
        # Ash Color Recipe
        # Red: Band 15 - Band 13
        r = self._normalize_range(b15 - b13, -4, 2)

        # Green: Band 14 - Band 11
        g = self._normalize_range(b14 - b11, -4, 5)

        # Blue: Band 13
        b = self._normalize_range(b13, 243, 303)

        # Stack and clip
        ash = np.stack([r, g, b], axis=-1)
        return np.clip(ash, 0, 1)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        record_id = str(row["record_id"])

        # Define band IDs
        band_ids = [11, 13, 14, 15]
        bands = {}

        # Load required bands
        # NPY shape is (H, W, T). T=8 usually.
        # Labeled frame is at index 4 (5th frame). Previous is index 3.
        t_curr = 4
        t_prev = 3

        for bid in band_ids:
            col_name = f"band_{bid:02d}"
            file_path = os.path.join(Config.INPUT_DIR, row[col_name])
            try:
                data = np.load(file_path)
                bands[bid] = data
            except Exception as e:
                # Fallback for missing files (should not happen in clean data)
                # Create dummy data of correct shape
                bands[bid] = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 8), dtype=np.float32
                )

        # Extract frames for current and previous time steps
        # Using .astype(float) to ensure precision during subtraction
        b11_t = bands[11][..., t_curr]
        b13_t = bands[13][..., t_curr]
        b14_t = bands[14][..., t_curr]
        b15_t = bands[15][..., t_curr]

        b11_prev = bands[11][..., t_prev]
        b13_prev = bands[13][..., t_prev]
        b14_prev = bands[14][..., t_prev]
        b15_prev = bands[15][..., t_prev]

        # Compute Ash Color
        ash_t = self._get_ash_color(b11_t, b13_t, b14_t, b15_t)
        ash_prev = self._get_ash_color(b11_prev, b13_prev, b14_prev, b15_prev)

        # Compute Temporal Difference
        ash_diff = ash_t - ash_prev

        # Construct 6-channel input: [Ash_t, Ash_Diff]
        # Shape: (H, W, 6)
        img = np.concatenate([ash_t, ash_diff], axis=-1)

        # Load Mask if available (Train/Val)
        mask = None
        if self.split in ["train", "validation"]:
            mask_col = "human_pixel_masks"
            if mask_col in row and pd.notna(row[mask_col]):
                mask_path = os.path.join(Config.INPUT_DIR, row[mask_col])
                try:
                    # Mask shape: (H, W, 1)
                    mask = np.load(mask_path).astype(np.float32)
                except:
                    mask = np.zeros(
                        (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1), dtype=np.float32
                    )
            else:
                mask = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1), dtype=np.float32
                )

        # Apply Augmentations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
                # Ensure mask is (1, H, W) after ToTensorV2 (which might make it H,W if squeezed)
                # ToTensorV2 usually keeps channel dim if it exists in input and is handled correctly.
                # However, for masks, sometimes it's better to be explicit.
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.ndim == 3 and mask.shape[0] != 1:
                    # If albumentations returned (H, W, 1) -> permute to (1, H, W)
                    # But ToTensorV2 converts HWC to CHW.
                    pass
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]

        # Return dictionary or tuple compatible with DataLoader
        if self.split in ["train", "validation"]:
            return img.float(), mask.float()
        else:
            return img.float(), record_id
