import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import normalize_image


def load_fragment_slab(fragment_id, volume_dir, z_start, z_end, load_cached_data=True):
    """
    Loads a specific Z-slab of the 3D volume for a fragment.
    Uses caching to speed up subsequent loads.

    Args:
        fragment_id: ID of the fragment.
        volume_dir: Relative path to the volume directory.
        z_start: Start index of the Z-slice.
        z_end: End index of the Z-slice (exclusive).
        load_cached_data: Whether to attempt loading from cache.

    Returns:
        Numpy array of shape (D, H, W) containing the volume slab.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Construct cache filename
    cache_path = os.path.join(
        cache_dir, f"frag_{fragment_id}_slab_{z_start}_{z_end}.npy"
    )

    if load_cached_data and os.path.exists(cache_path):
        try:
            volume = np.load(cache_path)
            # Basic integrity check: depth must match requested range
            if volume.shape[0] == (z_end - z_start):
                return volume
            else:
                print(f"Cached volume shape mismatch for {fragment_id}. Reloading.")
        except Exception as e:
            print(f"Error loading cache for {fragment_id}: {e}. Reloading.")

    # Load from source TIFFs
    slices = []
    # z_range is [z_start, z_end)
    for z in range(z_start, z_end):
        filename = f"{z:02d}.tif"
        # volume_dir is relative to INPUT_DIR
        path = os.path.join(Config.INPUT_DIR, volume_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Slice {path} not found.")

        # Load image (uint16)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        slices.append(img)

    volume = np.stack(slices, axis=0)  # (D, H, W)

    # Save to cache
    try:
        np.save(cache_path, volume)
    except Exception as e:
        print(f"Failed to save cache for {fragment_id}: {e}")

    return volume


class InkDataset(Dataset):
    def __init__(
        self,
        metadata,
        z_start,
        z_end,
        mode="train",
        transform=None,
        load_cached_data=True,
    ):
        """
        Args:
            metadata: DataFrame or path to CSV.
            z_start: Start Z-index for the specialist window.
            z_end: End Z-index for the specialist window.
            mode: 'train', 'validation', or 'test'.
            transform: Albumentations transform.
            load_cached_data: Whether to use cached .npy volumes.
        """
        self.z_start = z_start
        self.z_end = z_end
        self.mode = mode

        # Load metadata
        if isinstance(metadata, str):
            self.df = pd.read_csv(metadata)
        else:
            self.df = metadata.copy()

        # Handle Test Mode (Tiling)
        # If 'x' and 'y' columns are missing (as in test.csv), we generate tiles
        if "x" not in self.df.columns or "y" not in self.df.columns:
            self.df = self._expand_metadata_with_tiles(self.df)

        # Pre-load volumes into memory
        # We identify unique fragments and load their slabs once
        self.volumes = {}
        self.masks = {}  # Cache full masks to avoid re-reading
        self.labels = {}  # Cache full labels

        # Ensure fragment_id is string for consistency
        self.df["fragment_id"] = self.df["fragment_id"].astype(str)
        unique_frags = self.df["fragment_id"].unique()

        for fid in unique_frags:
            # Get volume path from the first occurrence
            row = self.df[self.df["fragment_id"] == fid].iloc[0]
            vol_path = row["volume_path"]

            # Load Volume Slab
            self.volumes[fid] = load_fragment_slab(
                fid, vol_path, z_start, z_end, load_cached_data
            )

            # Load Masks/Labels (Full size)
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            self.masks[fid] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Load label if available (Train/Val)
            if mode != "test":
                if "label_path" in row and pd.notna(row["label_path"]):
                    label_path = os.path.join(Config.INPUT_DIR, row["label_path"])
                    self.labels[fid] = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                else:
                    # Fallback if label missing
                    self.labels[fid] = np.zeros_like(self.masks[fid])

        # Define default transforms
        if transform is None:
            if self.mode == "train":
                self.transform = A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose([ToTensorV2()])
        else:
            self.transform = transform

    def _expand_metadata_with_tiles(self, df):
        """
        Expands fragment-level metadata into patch-level metadata for inference.
        Generates sliding window coordinates.
        """
        new_rows = []
        stride = Config.INFERENCE_STRIDE
        size = Config.TILE_SIZE

        for _, row in df.iterrows():
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])

            # Read mask to get dimensions
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            h, w = mask.shape

            # Generate coordinates
            y_points = range(0, h, stride)
            x_points = range(0, w, stride)

            for y in y_points:
                for x in x_points:
                    item = row.to_dict()
                    item["x"] = x
                    item["y"] = y
                    item["width"] = size
                    item["height"] = size
                    new_rows.append(item)

        return pd.DataFrame(new_rows)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fid = str(row["fragment_id"])
        x, y = int(row["x"]), int(row["y"])
        w, h = int(row["width"]), int(row["height"])

        # 1. Get Volume Crop
        # Volume shape: (D, H_full, W_full)
        full_vol = self.volumes[fid]
        vol_d, vol_h, vol_w = full_vol.shape

        # Calculate crop coordinates
        x_end = min(x + w, vol_w)
        y_end = min(y + h, vol_h)

        # Crop volume
        vol_crop = full_vol[:, y:y_end, x:x_end]

        # Pad if necessary (at edges)
        pad_h = h - vol_crop.shape[1]
        pad_w = w - vol_crop.shape[2]

        if pad_h > 0 or pad_w > 0:
            vol_crop = np.pad(
                vol_crop,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

        # 2. Projection (Overlapping Thick Slab)
        # We split the 24-slice volume into 3 chunks of 12 slices with 6 overlap
        # Channel 0: 0-12
        # Channel 1: 6-18
        # Channel 2: 12-24

        ch1 = np.max(vol_crop[0:12], axis=0)
        ch2 = np.max(vol_crop[6:18], axis=0)
        ch3 = np.max(vol_crop[12:24], axis=0)

        image = np.stack([ch1, ch2, ch3], axis=-1)  # (H, W, 3)

        # 3. Normalization
        # Convert uint16 to float [0, 1]
        image = image.astype(np.float32) / 65535.0
        # Apply ImageNet normalization
        image = normalize_image(image)

        # 4. Get Masks
        full_mask = self.masks[fid]
        mask_crop = full_mask[y:y_end, x:x_end]

        if self.mode != "test":
            full_label = self.labels[fid]
            label_crop = full_label[y:y_end, x:x_end]
        else:
            label_crop = np.zeros_like(mask_crop)

        # Pad masks if necessary
        if pad_h > 0 or pad_w > 0:
            mask_crop = np.pad(
                mask_crop, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
            )
            label_crop = np.pad(
                label_crop, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
            )

        # Binarize
        mask_crop = (mask_crop > 0).astype(np.float32)
        label_crop = (label_crop > 0).astype(np.float32)

        # 5. Augmentation
        # Albumentations expects HWC for image
        # We pass label as first mask, validity mask as second
        augmented = self.transform(image=image, masks=[label_crop, mask_crop])
        image = augmented["image"]
        label_crop = augmented["masks"][0]
        mask_crop = augmented["masks"][1]

        # Ensure masks are (1, H, W)
        if label_crop.ndim == 2:
            label_crop = label_crop.unsqueeze(0)
        if mask_crop.ndim == 2:
            mask_crop = mask_crop.unsqueeze(0)

        return {
            "image": image,
            "label": label_crop,
            "valid_mask": mask_crop,
            "fragment_id": fid,
            "x": x,
            "y": y,
        }
