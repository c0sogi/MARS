import os
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from library.config import Config


class InkDataset(Dataset):
    """
    Dataset for loading 3D X-ray volume crops and corresponding ink labels.
    Handles loading of 65-slice volumes, normalization, and caching.
    """

    def __init__(
        self, metadata_df, mode="train", load_cached_data=True, transform=None
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing patch metadata.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use disk caching for processed volumes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = metadata_df
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.transform = transform

        # Define cache directory inside the working directory
        # We use a subdirectory to keep things organized
        self.cache_dir = os.path.join(Config.WORKING_DIR, "dataset_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.metadata)

    def _load_volume(self, row):
        """
        Loads the 3D volume for the patch. Implements the required caching logic.
        Returns:
            np.ndarray: 5D array of shape (1, Z_DIM, H, W) normalized to [0, 1].
        """
        sample_id = row["sample_id"]
        # Construct a unique filename for the cache
        cache_filename = f"{sample_id}_vol.npy"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        # Target dimensions from Config (ensures divisibility by 32)
        target_h, target_w = Config.PATCH_SIZE

        # 1. Try to load from cache if enabled
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                volume = np.load(cache_path)
                # Check if cached volume matches the required fixed size
                if volume.shape[-2:] == (target_h, target_w):
                    return volume
            except Exception:
                # If loading fails (corrupt file), proceed to compute from scratch
                pass

        # 2. Compute from scratch
        surface_vol_rel_path = row["surface_volume_path"]
        x, y = row["x"], row["y"]
        valid_w, valid_h = row["w"], row["h"]

        # Construct full path to the surface volume directory
        # Config.INPUT_DIR is the root, surface_vol_rel_path is like "train/1/surface_volume"
        vol_dir = os.path.join(Config.INPUT_DIR, surface_vol_rel_path)

        slices = []
        # Iterate through all Z slices
        for z in range(Config.Z_DIM):
            # Filenames are formatted as 00.tif, 01.tif, ...
            filename = f"{z:02d}.tif"
            file_path = os.path.join(vol_dir, filename)

            # Initialize slice container with FIXED size (Padding)
            img_crop = np.zeros((target_h, target_w), dtype=np.float32)

            if os.path.exists(file_path):
                # Load as grayscale (0-255)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Crop the patch
                    # Ensure we don't go out of bounds (though metadata should be correct)
                    img_h, img_w = img.shape
                    curr_valid_h = min(valid_h, img_h - y)
                    curr_valid_w = min(valid_w, img_w - x)

                    if curr_valid_h > 0 and curr_valid_w > 0:
                        img_crop[:curr_valid_h, :curr_valid_w] = img[
                            y : y + curr_valid_h, x : x + curr_valid_w
                        ].astype(np.float32)

            slices.append(img_crop)

        # Stack slices along the depth dimension -> (Depth, Height, Width)
        volume = np.stack(slices, axis=0)

        # Normalize to [0, 1]
        volume = volume / 255.0

        # Add Channel dimension -> (1, Depth, Height, Width)
        volume = np.expand_dims(volume, axis=0)

        # Save to cache for future runs
        try:
            np.save(cache_path, volume)
        except Exception:
            # Ignore write errors (e.g., disk full or concurrency issues)
            pass

        return volume

    def _load_mask(self, row):
        """
        Loads the binary mask (ink label).
        Returns:
            np.ndarray: 3D array of shape (1, H, W).
        """
        # Target dimensions
        target_h, target_w = Config.PATCH_SIZE

        # Initialize with FIXED size
        mask_crop = np.zeros((target_h, target_w), dtype=np.float32)

        # For test mode, return a dummy mask with correct shape
        if self.mode == "test":
            return np.expand_dims(mask_crop, axis=0)

        ink_path = row.get("inklabels_path")

        # If no label path is provided or it is NaN
        if pd.isna(ink_path):
            return np.expand_dims(mask_crop, axis=0)

        full_path = os.path.join(Config.INPUT_DIR, ink_path)

        if not os.path.exists(full_path):
            return np.expand_dims(mask_crop, axis=0)

        # Load image
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.expand_dims(mask_crop, axis=0)

        # Crop
        x, y = row["x"], row["y"]
        valid_w, valid_h = row["w"], row["h"]

        img_h, img_w = img.shape
        curr_valid_h = min(valid_h, img_h - y)
        curr_valid_w = min(valid_w, img_w - x)

        if curr_valid_h > 0 and curr_valid_w > 0:
            crop = img[y : y + curr_valid_h, x : x + curr_valid_w]
            # Binarize: Ink is present if pixel > 0
            # Place into padded buffer
            mask_crop[:curr_valid_h, :curr_valid_w] = (crop > 0).astype(np.float32)

        # Add Channel dimension -> (1, Height, Width)
        mask_crop = np.expand_dims(mask_crop, axis=0)

        return mask_crop

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load data
        volume = self._load_volume(row)
        mask = self._load_mask(row)

        # Convert to torch tensors
        volume_tensor = torch.from_numpy(volume)
        mask_tensor = torch.from_numpy(mask)

        sample = {
            "image": volume_tensor,
            "mask": mask_tensor,
            "sample_id": row["sample_id"],
            "fragment_id": str(row["fragment_id"]),
            "x": row["x"],
            "y": row["y"],
            "w": row["w"],
            "h": row["h"],
        }

        # Apply transforms if provided
        # Note: Since the input is 5D (1, D, H, W) and mask is 3D (1, H, W),
        # standard 2D transforms need careful application.
        # We assume the user-provided transform handles this dictionary structure if given.
        if self.transform:
            sample = self.transform(sample)

        return sample


def get_dataset(split="train", load_cached_data=True):
    """
    Factory function to create datasets based on configuration.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to enable disk caching.

    Returns:
        InkDataset: The initialized dataset.
    """
    if split == "train":
        df = pd.read_csv(Config.TRAIN_METADATA)
        mode = "train"
    elif split == "val":
        df = pd.read_csv(Config.VAL_METADATA)
        mode = "val"
    elif split == "test":
        df = pd.read_csv(Config.TEST_METADATA)
        mode = "test"
    else:
        raise ValueError(f"Unknown split: {split}")

    return InkDataset(df, mode=mode, load_cached_data=load_cached_data)
