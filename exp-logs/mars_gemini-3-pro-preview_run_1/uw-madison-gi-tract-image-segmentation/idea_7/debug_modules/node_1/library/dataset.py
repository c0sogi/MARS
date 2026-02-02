import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    IMG_SIZE,
    SEQ_LEN,
    LOWER_PERCENTILE,
    UPPER_PERCENTILE,
    CLASSES,
    WORKING_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)
from library.utils import rle_decode, percentile_normalize


def process_volume(group_df, input_dir, img_size):
    """
    Loads, resizes, and normalizes all slices for a specific Case-Day group.
    Returns stacked numpy arrays for images, masks, and IDs.
    """
    # Ensure slices are ordered spatially
    group_df = group_df.sort_values("slice")

    images = []
    masks = []
    ids = []

    # Check if mask columns exist (Train/Val vs Test)
    has_masks = "large_bowel" in group_df.columns

    for _, row in group_df.iterrows():
        # --- Process Image ---
        img_path = os.path.join(input_dir, row["file_path"])
        # Load as-is (handling potential 16-bit depth correctly by not forcing 8-bit)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            # Fallback for missing files (though metadata check passed)
            img = np.zeros(img_size, dtype=np.float32)

        # Resize (Bilinear for images)
        img = cv2.resize(
            img, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR
        )

        # Normalize
        img = percentile_normalize(img, LOWER_PERCENTILE, UPPER_PERCENTILE)
        images.append(img)

        ids.append(row["id"])

        # --- Process Masks ---
        if has_masks:
            slice_masks = []
            for cls in CLASSES:
                rle = row[cls]
                # Decode RLE to binary mask
                mask = rle_decode(rle, (row["height"], row["width"]))
                # Resize (Nearest Neighbor for masks to preserve binary values)
                mask = cv2.resize(
                    mask, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST
                )
                slice_masks.append(mask)

            # Stack classes channel-wise: (H, W, C)
            masks.append(np.stack(slice_masks, axis=-1))

    # Stack along depth dimension
    vol_imgs = np.stack(images, axis=0)  # Shape: (D, H, W)

    if has_masks:
        vol_masks = np.stack(masks, axis=0)  # Shape: (D, H, W, C)
    else:
        # Create a placeholder or None
        vol_masks = np.empty((0, img_size[0], img_size[1], len(CLASSES)))

    return {"images": vol_imgs, "masks": vol_masks, "ids": np.array(ids)}


class SliceSequenceDataset(Dataset):
    def __init__(self, df, mode="train", cache_dir=None, load_cached_data=True):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
            cache_dir (str): Directory to store processed .npz files.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        self.df = df
        self.mode = mode
        self.cache_dir = cache_dir if cache_dir else os.path.join(WORKING_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.volume_cache = {}
        self.indices = []

        # Group metadata by (Case, Day) to form volumes
        groups = df.groupby(["case", "day"])

        for (case, day), group in groups:
            cache_file = os.path.join(self.cache_dir, f"{case}_{day}.npz")

            loaded = False
            data_dict = {}

            # 1. Try Loading Cache
            if load_cached_data and os.path.exists(cache_file):
                try:
                    with np.load(cache_file) as data:
                        # Load into memory
                        data_dict["images"] = data["images"]
                        data_dict["masks"] = data["masks"]
                        data_dict["ids"] = data["ids"]
                    loaded = True
                except Exception as e:
                    print(f"Failed to load cache for {case}_{day}: {e}")
                    loaded = False

            # 2. Process if not loaded
            if not loaded:
                processed = process_volume(group, INPUT_DIR, IMG_SIZE)
                data_dict = processed
                # Save to cache
                np.savez(
                    cache_file,
                    images=processed["images"],
                    masks=processed["masks"],
                    ids=processed["ids"],
                )

            # 3. Store in RAM (Dataset is kept in memory for speed)
            # Key is unique identifier for the volume
            key = f"{case}_{day}"
            self.volume_cache[key] = data_dict

            # 4. Build Index
            # Map global index to (Volume Key, Slice Index)
            # This allows __getitem__ to access any slice in the dataset
            num_slices = len(data_dict["ids"])
            for i in range(num_slices):
                self.indices.append(
                    {"key": key, "slice_idx": i, "id": data_dict["ids"][i]}
                )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        info = self.indices[idx]
        key = info["key"]
        center_idx = info["slice_idx"]

        data = self.volume_cache[key]
        vol_imgs = data["images"]  # (D, H, W)
        vol_masks = data["masks"]  # (D, H, W, C) or empty

        D, H, W = vol_imgs.shape

        # --- Construct Sequence ---
        # We want a sequence of length SEQ_LEN centered at center_idx
        # e.g., if SEQ_LEN=5, offsets: -2, -1, 0, 1, 2
        half_seq = SEQ_LEN // 2
        seq_indices = []

        for offset in range(-half_seq, half_seq + 1):
            i = center_idx + offset
            # Handle Boundary: Edge Replication (Clamp index)
            i = max(0, min(D - 1, i))
            seq_indices.append(i)

        # Extract sequence images: (Seq, H, W)
        seq_imgs = vol_imgs[seq_indices]

        # --- Format Input Tensor ---
        # ResNet backbone expects 3 channels.
        # We replicate the grayscale image to 3 channels.
        # Shape: (Seq, H, W) -> (Seq, 1, H, W) -> (Seq, 3, H, W)
        seq_imgs = np.expand_dims(seq_imgs, axis=1)
        seq_imgs = np.repeat(seq_imgs, 3, axis=1)

        img_tensor = torch.from_numpy(seq_imgs).float()

        # --- Format Target Mask ---
        # We only predict the mask for the CENTRAL slice
        if self.mode in ["train", "val"] and vol_masks.shape[0] > 0:
            # Extract central mask: (H, W, C)
            mask = vol_masks[center_idx]

            # PyTorch expects (C, H, W)
            mask = np.moveaxis(mask, -1, 0)
            mask_tensor = torch.from_numpy(mask).float()

            return img_tensor, mask_tensor
        else:
            # Test mode: Return ID for submission file generation
            return img_tensor, info["id"]


def get_dataloaders(train_df, val_df, test_df=None):
    """
    Factory function to create DataLoaders.
    """
    loaders = {}

    if train_df is not None:
        train_ds = SliceSequenceDataset(train_df, mode="train")
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    if val_df is not None:
        val_ds = SliceSequenceDataset(val_df, mode="val")
        loaders["val"] = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    if test_df is not None:
        test_ds = SliceSequenceDataset(test_df, mode="test")
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    return loaders
