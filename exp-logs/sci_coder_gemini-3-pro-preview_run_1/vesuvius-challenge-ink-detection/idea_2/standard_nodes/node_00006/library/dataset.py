import os
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_patch_volume


class InkDataset(Dataset):
    def __init__(self, mode="train", limit=None, load_cached_data=True):
        """
        PyTorch Dataset for Ink Detection from 3D X-ray scans.

        Args:
            mode (str): One of 'train', 'val', 'test'.
            limit (int, optional): Limit the number of samples (for debugging).
            load_cached_data (bool): Whether to load/save processed data from disk cache.
        """
        self.mode = mode
        self.limit = limit
        self.load_cached_data = load_cached_data

        # Determine which metadata file to use
        if self.mode == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
        elif self.mode == "val":
            self.metadata_path = Config.VAL_METADATA_PATH
        elif self.mode == "test":
            self.metadata_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be 'train', 'val', or 'test'."
            )

        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

        # Apply limit if requested
        if self.limit is not None:
            self.df = self.df.iloc[: self.limit].reset_index(drop=True)

        # Ensure the checkpoint/cache directory exists
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]

        # Define cache filenames
        vol_cache_path = os.path.join(Config.CHECKPOINT_DIR, f"{sample_id}_vol.npy")
        mask_cache_path = os.path.join(Config.CHECKPOINT_DIR, f"{sample_id}_mask.npy")

        volume = None
        label = None

        # --- 1. Try Loading from Cache ---
        if self.load_cached_data:
            if os.path.exists(vol_cache_path):
                try:
                    volume = np.load(vol_cache_path)

                    # For train/val, we also need the mask
                    if self.mode != "test":
                        if os.path.exists(mask_cache_path):
                            label = np.load(mask_cache_path)
                        else:
                            # If mask is missing but volume exists, force re-processing
                            volume = None
                except Exception:
                    # If loading fails (corrupt file), force re-processing
                    volume = None

        # --- 2. Process from Scratch (if not in cache) ---
        if volume is None:
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]

            # Load 3D Volume (Z, h, w)
            volume = load_patch_volume(row["surface_volume_path"], x, y, w, h)

            # Pad Volume to PATCH_SIZE if necessary
            # volume shape is (Z, H, W). We pad H and W.
            pad_h = Config.PATCH_SIZE - h
            pad_w = Config.PATCH_SIZE - w

            if pad_h > 0 or pad_w > 0:
                volume = np.pad(
                    volume,
                    ((0, 0), (0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

            # Normalize Volume
            volume = volume.astype(np.float32) / 255.0

            # Save Volume to Cache
            np.save(vol_cache_path, volume)

            # Load Label (only for train/val)
            if self.mode != "test":
                ink_path = row.get("inklabels_path")
                label_crop = None

                if pd.notna(ink_path):
                    full_ink_path = os.path.join(Config.INPUT_DIR, ink_path)
                    if os.path.exists(full_ink_path):
                        # Load full image and crop
                        # Note: Loading full image is memory intensive but necessary without
                        # specialized libraries like rasterio for windowed reading.
                        # Given the constraints and caching strategy, this is acceptable for the first run.
                        img = cv2.imread(full_ink_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            label_crop = img[y : y + h, x : x + w]

                # Handle cases where label loading failed or path was NaN
                if label_crop is None:
                    label_crop = np.zeros((h, w), dtype=np.uint8)

                # Pad Label
                if pad_h > 0 or pad_w > 0:
                    label_crop = np.pad(
                        label_crop,
                        ((0, pad_h), (0, pad_w)),
                        mode="constant",
                        constant_values=0,
                    )

                # Binarize and Normalize Label
                label = (label_crop > 0).astype(np.float32)

                # Save Label to Cache
                np.save(mask_cache_path, label)

        # --- 3. Convert to Tensors ---
        # Volume: (Z, H, W)
        volume_tensor = torch.from_numpy(volume)

        if self.mode != "test":
            # Label: (H, W) -> (1, H, W)
            # Add channel dimension for BCEWithLogitsLoss
            label_tensor = torch.from_numpy(label).unsqueeze(0)
            return volume_tensor, label_tensor
        else:
            return volume_tensor
