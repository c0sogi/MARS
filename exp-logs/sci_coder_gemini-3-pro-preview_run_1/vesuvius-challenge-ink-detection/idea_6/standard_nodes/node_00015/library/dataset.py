import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class InkDataset(Dataset):
    """
    Dataset for loading 3D X-ray volume slices and corresponding ink labels.
    Handles caching of processed volumes to speed up training.
    """

    def __init__(self, metadata_file, load_cached_data=True, transform=None):
        """
        Args:
            metadata_file (str): Path to the metadata CSV file.
            load_cached_data (bool): Whether to use cached .npy files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(metadata_file)
        self.load_cached_data = load_cached_data
        self.transform = transform

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _get_cache_path(self, fragment_id, y, x, suffix):
        """Generates the cache file path for a specific patch."""
        filename = f"{fragment_id}_{y}_{x}_{suffix}.npy"
        return os.path.join(Config.CACHE_DIR, filename)

    def load_volume(self, row):
        """
        Loads the 3D volume (65 slices).
        Checks cache first, otherwise loads from TIFFs, normalizes, and caches.
        """
        fragment_id = str(row["fragment_id"])
        y, x, w, h = row["y"], row["x"], row["w"], row["h"]

        cache_path = self._get_cache_path(fragment_id, y, x, "vol")

        # 1. Try loading from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                volume = np.load(cache_path)
                return volume
            except Exception:
                # If load fails (e.g. corrupt file), proceed to process from scratch
                pass

        # 2. Process from scratch
        slices = []
        # Construct base path for surface volume (relative to INPUT_DIR)
        base_path = os.path.join(Config.INPUT_DIR, row["surface_volume_path"])

        for i in range(Config.Z_DIM):
            filename = f"{i:02d}.tif"
            file_path = os.path.join(base_path, filename)

            if os.path.exists(file_path):
                # Load grayscale image
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    # Fallback for corrupt/unreadable image
                    patch = np.zeros((h, w), dtype=np.uint8)
                else:
                    # Crop to the specific patch
                    patch = img[y : y + h, x : x + w]
            else:
                # Handle missing file
                patch = np.zeros((h, w), dtype=np.uint8)

            slices.append(patch)

        # Stack slices: (65, h, w)
        volume = np.stack(slices, axis=0)

        # Normalize pixel values to [0, 1]
        volume = volume.astype(np.float32) / 255.0

        # 3. Save to cache
        try:
            np.save(cache_path, volume)
        except Exception as e:
            print(f"Warning: Failed to save volume cache to {cache_path}: {e}")

        return volume

    def load_label(self, row):
        """
        Loads the binary ink label.
        Checks cache first, otherwise loads from image, binarizes, and caches.
        """
        # Check if label path is valid (it won't be for test set)
        if "inklabels_path" not in row or pd.isna(row["inklabels_path"]):
            return None

        fragment_id = str(row["fragment_id"])
        y, x, w, h = row["y"], row["x"], row["w"], row["h"]

        cache_path = self._get_cache_path(fragment_id, y, x, "mask")

        # 1. Try loading from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                mask = np.load(cache_path)
                return mask
            except Exception:
                pass

        # 2. Process from scratch
        file_path = os.path.join(Config.INPUT_DIR, row["inklabels_path"])

        if os.path.exists(file_path):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                mask = np.zeros((h, w), dtype=np.float32)
            else:
                crop = img[y : y + h, x : x + w]
                # Binarize: Ink is 1, No-Ink is 0
                mask = (crop > 0).astype(np.float32)
        else:
            mask = np.zeros((h, w), dtype=np.float32)

        # 3. Save to cache
        try:
            np.save(cache_path, mask)
        except Exception as e:
            print(f"Warning: Failed to save mask cache to {cache_path}: {e}")

        return mask

    def __getitem__(self, idx):
        """
        Returns:
            volume_tensor (torch.Tensor): Shape (65, 512, 512)
            label_tensor (torch.Tensor): Shape (1, 512, 512)
        """
        row = self.df.iloc[idx]

        # Load raw data (may have variable H, W at edges)
        volume = self.load_volume(row)  # Shape: (65, h, w)
        label = self.load_label(row)  # Shape: (h, w) or None

        # Determine padding to reach fixed patch size
        _, h, w = volume.shape
        target_h, target_w = Config.PATCH_HEIGHT, Config.PATCH_WIDTH

        pad_h = target_h - h
        pad_w = target_w - w

        # Apply padding if necessary (at the bottom and right)
        if pad_h > 0 or pad_w > 0:
            # Pad volume (C, H, W) -> pad last two dimensions
            volume = np.pad(
                volume,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

            if label is not None:
                label = np.pad(
                    label, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
                )

        # Handle missing label (Test set) by creating a dummy zero mask
        if label is None:
            label = np.zeros((target_h, target_w), dtype=np.float32)

        # Convert to PyTorch Tensors
        volume_tensor = torch.from_numpy(volume)

        # Add channel dimension to label for Loss compatibility: (H, W) -> (1, H, W)
        label_tensor = torch.from_numpy(label).unsqueeze(0)

        if self.transform:
            # Note: Custom transforms would need to handle 3D volume and 2D mask appropriately
            pass

        return volume_tensor, label_tensor
