import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config

# Global cache to hold loaded volumes in memory during the process lifetime
_MEMORY_VOLUME_CACHE = {}


def get_volume(fragment_id, volume_dir, load_cached_data=True):
    """
    Retrieves the 3D volume for a specific fragment.
    Implements a two-level cache:
    1. Memory Cache: Returns object if already loaded in RAM.
    2. Disk Cache: Checks for .npy file in WORKING_DIR.
    3. Raw Load: Loads .tif files from INPUT_DIR, saves .npy, returns volume.

    Loads slices [16, 48) (exclusive) to cover all Views A, B, C.
    """
    global _MEMORY_VOLUME_CACHE

    if fragment_id in _MEMORY_VOLUME_CACHE:
        return _MEMORY_VOLUME_CACHE[fragment_id]

    # Define the global Z-range required for all views
    # Min start is View A (16).
    # Max end is View C start (24) + Ch2 offset (12) + Thickness (12) = 48.
    z_start = 16
    z_end = 48
    num_slices = z_end - z_start

    cache_dir = os.path.join(Config.WORKING_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{fragment_id}_vol_{z_start}_{z_end}.npy")

    # 1. Try Disk Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            volume = np.load(cache_path)
            _MEMORY_VOLUME_CACHE[fragment_id] = volume
            return volume
        except Exception as e:
            print(
                f"Failed to load cached volume for {fragment_id}: {e}. Reloading from source."
            )

    # 2. Load from Raw Tifs
    # We need to determine the dimensions first
    # Load the first slice to get H, W
    first_slice_path = os.path.join(Config.INPUT_DIR, volume_dir, f"{z_start:02d}.tif")
    if not os.path.exists(first_slice_path):
        raise FileNotFoundError(f"Slice {first_slice_path} not found.")

    img0 = cv2.imread(first_slice_path, cv2.IMREAD_UNCHANGED)
    h, w = img0.shape

    # Pre-allocate volume
    volume = np.zeros((num_slices, h, w), dtype=np.uint16)

    for i, z in enumerate(range(z_start, z_end)):
        path = os.path.join(Config.INPUT_DIR, volume_dir, f"{z:02d}.tif")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            volume[i] = img
        else:
            # If a slice is missing (rare), leave as zeros or handle error
            # Assuming data integrity based on prompt
            pass

    # 3. Save to Disk Cache
    np.save(cache_path, volume)

    _MEMORY_VOLUME_CACHE[fragment_id] = volume
    return volume


class InkDataset(Dataset):
    def __init__(self, mode="train", limit=None, load_cached_data=True):
        """
        Args:
            mode (str): 'train' or 'validation'.
            limit (int): Limit dataset size for debugging.
            load_cached_data (bool): Use cached .npy volumes.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_METADATA_PATH)
            # Augmentations for training
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    ToTensorV2(),
                ]
            )
        else:
            self.df = pd.read_csv(Config.VALID_METADATA_PATH)
            # No geometric augmentations for validation
            self.transform = A.Compose([ToTensorV2()])

        if limit:
            self.df = self.df.iloc[:limit].reset_index(drop=True)

        # Pre-load all volumes referenced in metadata
        self.fragment_ids = self.df["fragment_id"].unique()
        self.volumes = {}
        self.masks = {}
        self.labels = {}

        # We also need to load full masks and labels for patch extraction context if needed,
        # but the metadata already provides x, y. We load masks/labels to extract patches.
        # Actually, reading small patches from disk pngs is slow.
        # Better to cache the full mask/label images in memory too.

        for fid in self.fragment_ids:
            row = self.df[self.df["fragment_id"] == fid].iloc[0]
            vol_path = row["volume_path"]
            mask_path = row["mask_path"]
            label_path = row["label_path"]

            # Load Volume
            self.volumes[fid] = get_volume(fid, vol_path, self.load_cached_data)

            # Load Mask
            m_path = os.path.join(Config.INPUT_DIR, mask_path)
            self.masks[fid] = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)

            # Load Label
            l_path = os.path.join(Config.INPUT_DIR, label_path)
            self.labels[fid] = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fid = row["fragment_id"]
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        # 1. Select View
        # Train: Random View. Val: Fixed View B (Center).
        if self.mode == "train":
            view_name = np.random.choice(list(Config.VIEWS.keys()))
        else:
            view_name = "B"

        view_start_z = Config.VIEWS[view_name]

        # 2. Extract Input Volume Patch
        # The volume in memory starts at Z=16.
        # We need to adjust indices relative to this offset.
        # Config.get_channel_ranges returns absolute Z indices.
        # We subtract 16 to get local indices.

        vol_offset = 16
        channel_ranges = Config.get_channel_ranges(view_start_z)

        channels = []
        full_vol = self.volumes[fid]

        # Ensure patch is within bounds of the image
        # Metadata generation should guarantee this, but we clip to be safe
        img_h, img_w = self.masks[fid].shape
        y_end = min(y + h, img_h)
        x_end = min(x + w, img_w)

        # Calculate padding if patch is smaller than TILE_SIZE
        pad_h = Config.TILE_SIZE - (y_end - y)
        pad_w = Config.TILE_SIZE - (x_end - x)

        for start, end in channel_ranges:
            local_start = start - vol_offset
            local_end = end - vol_offset

            # Extract slab: (Thickness, H_patch, W_patch)
            slab = full_vol[local_start:local_end, y:y_end, x:x_end]

            # Projection: Mean
            # Result: (H_patch, W_patch)
            if slab.shape[0] > 0:
                projection = np.mean(slab, axis=0).astype(np.float32)
            else:
                projection = np.zeros((y_end - y, x_end - x), dtype=np.float32)

            channels.append(projection)

        # Stack: (H, W, 3)
        image = np.stack(channels, axis=-1)

        # Normalize [0, 65535] -> [0, 1]
        image = image / 65535.0

        # 3. Extract Label Patch
        label_img = self.labels[fid][y:y_end, x:x_end]
        # Binarize label (0 or 255 -> 0 or 1)
        label = (label_img > 0).astype(np.float32)

        # 4. Padding if necessary (at edges)
        if pad_h > 0 or pad_w > 0:
            image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            label = np.pad(
                label, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
            )

        # 5. Augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        # Label needs to be (1, H, W) for BCE loss usually, or just (H, W) depending on implementation.
        # Albumentations ToTensorV2 returns mask as (H, W) if it's 2D.
        # We add channel dim for consistency with model output (B, 1, H, W)
        if label.ndim == 2:
            label = label.unsqueeze(0)

        return image, label


class TestInkDataset(Dataset):
    def __init__(self, fragment_id, view="B", load_cached_data=True):
        """
        Dataset for inference on a single fragment.
        Tiles the fragment into non-overlapping patches.

        Args:
            fragment_id (str): ID of the fragment to process.
            view (str): 'A', 'B', or 'C'.
            load_cached_data (bool): Use cached volumes.
        """
        self.fragment_id = fragment_id
        self.view = view

        # Load Test Metadata to find path
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)
        row = test_df[test_df["fragment_id"] == fragment_id].iloc[0]

        self.vol_path = row["volume_path"]
        self.mask_path = row["mask_path"]

        # Load Volume
        self.volume = get_volume(fragment_id, self.vol_path, load_cached_data)

        # Load Mask to get dimensions
        full_mask_path = os.path.join(Config.INPUT_DIR, self.mask_path)
        self.mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
        self.h, self.w = self.mask.shape

        # Generate Grid (Non-overlapping)
        self.tile_size = Config.TILE_SIZE
        self.coords = []
        for y in range(0, self.h, self.tile_size):
            for x in range(0, self.w, self.tile_size):
                self.coords.append((x, y))

        self.transform = A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]

        # Calculate crop dimensions
        y_end = min(y + self.tile_size, self.h)
        x_end = min(x + self.tile_size, self.w)

        # View Logic
        view_start_z = Config.VIEWS[self.view]
        vol_offset = 16
        channel_ranges = Config.get_channel_ranges(view_start_z)

        channels = []

        for start, end in channel_ranges:
            local_start = start - vol_offset
            local_end = end - vol_offset

            # Extract slab
            slab = self.volume[local_start:local_end, y:y_end, x:x_end]

            if slab.shape[0] > 0:
                projection = np.mean(slab, axis=0).astype(np.float32)
            else:
                projection = np.zeros((y_end - y, x_end - x), dtype=np.float32)

            channels.append(projection)

        image = np.stack(channels, axis=-1)
        image = image / 65535.0

        # Padding
        pad_h = self.tile_size - (y_end - y)
        pad_w = self.tile_size - (x_end - x)

        if pad_h > 0 or pad_w > 0:
            image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # Transform
        augmented = self.transform(image=image)
        image = augmented["image"]

        # Return coordinates and original size for reconstruction
        return (
            image,
            torch.tensor([x, y]),
            torch.tensor([self.w, self.h]),
            self.fragment_id,
        )
