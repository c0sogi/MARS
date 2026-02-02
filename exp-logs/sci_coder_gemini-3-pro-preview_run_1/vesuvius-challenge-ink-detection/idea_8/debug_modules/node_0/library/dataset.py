import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class InkDataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True, limit=None):
        """
        Dataset class for 3D Papyrus Ink Detection.

        Args:
            mode (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to use cached .npy files.
            limit (int, optional): Limit dataset size for debugging.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Select metadata file based on mode
        if mode == "train":
            self.csv_path = Config.TRAIN_CSV
        elif mode == "val":
            self.csv_path = Config.VAL_CSV
        elif mode == "test":
            self.csv_path = Config.TEST_CSV
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Load metadata
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        # Optional limit for debugging
        if limit is not None:
            self.df = self.df.iloc[:limit]

    def __len__(self):
        return len(self.df)

    def _get_cache_paths(self, sample_id):
        """Generates paths for cached numpy arrays."""
        vol_path = os.path.join(Config.CACHE_DIR, f"{sample_id}_vol.npy")
        mask_path = os.path.join(Config.CACHE_DIR, f"{sample_id}_mask.npy")
        label_path = os.path.join(Config.CACHE_DIR, f"{sample_id}_label.npy")
        return vol_path, mask_path, label_path

    def _process_from_disk(self, row):
        """
        Loads raw TIFF slices and masks from disk, stacks them, and normalizes.
        Returns numpy arrays.
        """
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]

        # 1. Load Volume (65 slices)
        # Construct path to surface volume directory
        vol_dir = os.path.join(Config.INPUT_DIR, row["surface_volume_path"])

        slices = []
        for i in range(Config.Z_DIM):
            # Filenames are 00.tif, 01.tif, ...
            filename = f"{i:02d}.tif"
            file_path = os.path.join(vol_dir, filename)

            # Load image
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Fallback: create empty if file missing (should not happen per verification)
                img = np.zeros((h, w), dtype=np.uint8)
            else:
                # Crop
                img = img[y : y + h, x : x + w]

            slices.append(img)

        # Stack slices: (Z, H, W)
        volume = np.stack(slices, axis=0)

        # Normalize to [0, 1] float32
        volume = volume.astype(np.float32) / 255.0

        # 2. Load Validity Mask
        mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            mask = np.zeros((h, w), dtype=np.float32)
        else:
            mask = mask_img[y : y + h, x : x + w]
            mask = (mask > 0).astype(np.float32)  # Binary 0.0 or 1.0

        # 3. Load Label (Ink) - only for train/val
        label = None
        if self.mode in ["train", "val"]:
            if "inklabels_path" in row and pd.notna(row["inklabels_path"]):
                ink_path = os.path.join(Config.INPUT_DIR, row["inklabels_path"])
                ink_img = cv2.imread(ink_path, cv2.IMREAD_GRAYSCALE)
                if ink_img is None:
                    label = np.zeros((h, w), dtype=np.float32)
                else:
                    label = ink_img[y : y + h, x : x + w]
                    label = (label > 0).astype(np.float32)
            else:
                # If no label path provided in metadata but mode is train/val
                label = np.zeros((h, w), dtype=np.float32)

        return volume, mask, label

    def process_sample(self, row, load_cached_data):
        """
        Retrieves data for a single sample, using cache if requested and available.
        Strictly follows the caching logic requirement.
        """
        sample_id = row["sample_id"]
        vol_cache_path, mask_cache_path, label_cache_path = self._get_cache_paths(
            sample_id
        )

        # Logic Step 1: Check cache if enabled
        if load_cached_data:
            # Check if files exist
            has_vol = os.path.exists(vol_cache_path)
            has_mask = os.path.exists(mask_cache_path)
            # For test mode, we don't check label cache
            has_label = (
                os.path.exists(label_cache_path)
                if self.mode in ["train", "val"]
                else True
            )

            if has_vol and has_mask and has_label:
                try:
                    volume = np.load(vol_cache_path)
                    mask = np.load(mask_cache_path)
                    if self.mode in ["train", "val"]:
                        label = np.load(label_cache_path)
                    else:
                        label = None
                    return volume, mask, label
                except Exception:
                    # Corrupt file, proceed to process from scratch
                    pass

        # Logic Step 2: Process from scratch
        volume, mask, label = self._process_from_disk(row)

        # Logic Step 3: Save to cache
        # Ensure directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        np.save(vol_cache_path, volume)
        np.save(mask_cache_path, mask)
        if label is not None:
            np.save(label_cache_path, label)

        return volume, mask, label

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Retrieve data using the caching logic
        volume, mask, label = self.process_sample(row, self.load_cached_data)

        # Convert to PyTorch tensors
        # Volume: (Z, H, W) -> FloatTensor
        volume_t = torch.from_numpy(volume).float()

        # Mask: (H, W) -> (1, H, W) -> FloatTensor
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        if label is not None:
            # Label: (H, W) -> (1, H, W) -> FloatTensor
            label_t = torch.from_numpy(label).unsqueeze(0).float()
        else:
            # For test set, return dummy label
            label_t = torch.zeros_like(mask_t)

        sample_id = row["sample_id"]

        return volume_t, label_t, mask_t, sample_id
