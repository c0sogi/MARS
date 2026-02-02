import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library import config


class InkDataset(Dataset):
    """
    Dataset for loading 3D X-ray volume slices and corresponding ink labels.
    Implements caching to .npy files to speed up training after the first epoch.
    """

    def __init__(self, df, mode="train", load_cached_data=True):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.cache_dir = config.CACHE_DIR
        self.target_h, self.target_w = config.PATCH_SIZE

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _pad_image(self, img):
        """
        Pads the image to (target_h, target_w) with zeros if it is smaller.
        Handles both 2D (H, W) and 3D (D, H, W) arrays.
        """
        if img.ndim == 2:
            h, w = img.shape
            pad_h = max(0, self.target_h - h)
            pad_w = max(0, self.target_w - w)
            if pad_h > 0 or pad_w > 0:
                return np.pad(
                    img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
                )
        elif img.ndim == 3:
            d, h, w = img.shape
            pad_h = max(0, self.target_h - h)
            pad_w = max(0, self.target_w - w)
            if pad_h > 0 or pad_w > 0:
                return np.pad(
                    img,
                    ((0, 0), (0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )
        return img

    def _load_volume(self, row):
        """
        Loads the 3D volume. Checks cache first.
        If not in cache, loads TIFFs, stacks, normalizes, and saves to cache.
        """
        sample_id = row["sample_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_id}_vol.npy")

        # 1. Try to load from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                volume = np.load(cache_path)
                return volume
            except Exception as e:
                # If load fails, proceed to recompute
                print(
                    f"Warning: Failed to load cache for {sample_id}, recomputing. Error: {e}"
                )

        # 2. Compute from scratch
        slices = []
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
        base_path = os.path.join(config.INPUT_DIR, row["surface_volume_path"])

        # Load all Z_DIM slices
        for i in range(config.Z_DIM):
            filename = f"{i:02d}.tif"
            file_path = os.path.join(base_path, filename)

            if os.path.exists(file_path):
                # Load as grayscale
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    # Fallback
                    crop = np.zeros((h, w), dtype=np.uint8)
                else:
                    crop = img[y : y + h, x : x + w]
            else:
                crop = np.zeros((h, w), dtype=np.uint8)

            slices.append(crop)

        # Stack to (65, h, w)
        volume = np.stack(slices, axis=0)

        # Pad to fixed patch size
        volume = self._pad_image(volume)

        # Normalize to [0, 1] and convert to float32
        volume = volume.astype(np.float32) / 255.0

        # 3. Save to cache
        try:
            np.save(cache_path, volume)
        except Exception as e:
            print(f"Warning: Failed to save cache for {sample_id}. Error: {e}")

        return volume

    def _load_label(self, row):
        """
        Loads the binary ink label.
        """
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
        path = os.path.join(config.INPUT_DIR, row["inklabels_path"])

        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                crop = np.zeros((h, w), dtype=np.uint8)
            else:
                crop = img[y : y + h, x : x + w]
        else:
            crop = np.zeros((h, w), dtype=np.uint8)

        # Pad to fixed patch size
        crop = self._pad_image(crop)

        # Binarize (0 or 1) and add channel dim -> (1, H, W)
        label = (crop > 0).astype(np.float32)
        return label[np.newaxis, ...]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load Volume
        volume = self._load_volume(row)
        volume_tensor = torch.from_numpy(volume)

        result = {
            "volume": volume_tensor,
            "sample_id": row["sample_id"],
            "fragment_id": row["fragment_id"],
            "x": row["x"],
            "y": row["y"],
        }

        # Load Label if training/validation
        if self.mode in ["train", "val"]:
            label = self._load_label(row)
            result["label"] = torch.from_numpy(label)

        return result


def get_dataloaders(
    batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, limit=None
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        limit (int, optional): If provided, limits the dataset size (for debugging).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(config.TRAIN_METADATA)
    val_df = pd.read_csv(config.VAL_METADATA)
    test_df = pd.read_csv(config.TEST_METADATA)

    # Apply limit if requested
    if limit is not None:
        train_df = train_df.iloc[:limit]
        val_df = val_df.iloc[:limit]
        test_df = test_df.iloc[:limit]
        print(f"Debug Mode: Limiting datasets to {limit} samples.")

    # Create Datasets
    train_ds = InkDataset(train_df, mode="train")
    val_ds = InkDataset(val_df, mode="val")
    test_ds = InkDataset(test_df, mode="test")

    # Create Dataloaders
    # Drop last for train to maintain stable batch statistics for BatchNorm
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
