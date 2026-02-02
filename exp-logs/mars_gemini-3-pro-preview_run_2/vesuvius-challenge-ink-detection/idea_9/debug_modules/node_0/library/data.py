import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG
from library.utils import seed_everything

# Constants for Z-slice loading
# We need to cover the max range defined in CFG.z_ranges (14-50) plus the jitter range (+/- 2).
# Range needed: [14-2, 50+2] = [12, 52].
# We load indices 12 up to 52 (exclusive of 52 in python slice, so 40 slices).
LOAD_Z_START = 12
LOAD_Z_END = 52


def load_fragment_data(
    fragment_id, volume_dir, mask_path, label_path=None, load_cached_data=True
):
    """
    Loads volume slices, mask, and label for a fragment.
    Caches the volume slices as a .npy file to speed up subsequent loads.
    """
    # Ensure working directory for cache exists
    os.makedirs(CFG.working_dir, exist_ok=True)

    # Define cache path
    cache_filename = f"fragment_{fragment_id}_slices_{LOAD_Z_START}_{LOAD_Z_END}.npy"
    cache_path = os.path.join(CFG.working_dir, cache_filename)

    volume = None

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            volume = np.load(cache_path)
        except Exception as e:
            print(f"Error loading cache for fragment {fragment_id}: {e}")
            volume = None

    # 2. If not in cache, load from TIFFs
    if volume is None:
        slices = []
        # Iterate through the required Z-indices
        for z in range(LOAD_Z_START, LOAD_Z_END):
            filename = f"{z:02d}.tif"
            path = os.path.join(CFG.input_dir, volume_dir, filename)

            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                slices.append(img)
            else:
                # Fallback for missing slices (should not happen based on dataset specs)
                # Create a zero slice matching the shape of a known slice (e.g., 32.tif)
                ref_path = os.path.join(CFG.input_dir, volume_dir, "32.tif")
                if os.path.exists(ref_path):
                    ref_img = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
                    slices.append(np.zeros_like(ref_img))
                else:
                    raise FileNotFoundError(
                        f"Reference slice 32.tif not found for {fragment_id}"
                    )

        # Stack into (D, H, W) array
        volume = np.stack(slices, axis=0)

        # Save to cache
        try:
            np.save(cache_path, volume)
        except Exception as e:
            print(f"Error saving cache for fragment {fragment_id}: {e}")

    # 3. Load Mask
    mask_full_path = os.path.join(CFG.input_dir, mask_path)
    mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        mask = (mask > 0).astype(np.uint8)

    # 4. Load Label (if provided)
    label = None
    if label_path:
        label_full_path = os.path.join(CFG.input_dir, label_path)
        if os.path.exists(label_full_path):
            label_img = cv2.imread(label_full_path, cv2.IMREAD_GRAYSCALE)
            if label_img is not None:
                label = (label_img > 0).astype(np.uint8)

    return volume, mask, label


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the given mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )


class InkDataset(Dataset):
    def __init__(self, metadata_df, mode="train", load_cached_data=True):
        self.mode = mode
        self.cfg = CFG
        self.samples = []
        self.fragments_data = {}  # In-memory storage for heavy volume data

        # 1. Load all referenced fragments into memory
        unique_frag_ids = metadata_df["fragment_id"].unique()

        for fid in unique_frag_ids:
            # Extract paths from the first row corresponding to this fragment
            row = metadata_df[metadata_df["fragment_id"] == fid].iloc[0]
            vol_path = row["volume_path"]
            mask_path = row["mask_path"]
            label_path = row.get("label_path", None)

            volume, mask, label = load_fragment_data(
                fid, vol_path, mask_path, label_path, load_cached_data
            )

            self.fragments_data[fid] = {"volume": volume, "mask": mask, "label": label}

        # 2. Prepare list of samples (patches)
        if mode in ["train", "validation"]:
            # For train/val, metadata contains specific patch coordinates
            self.samples = metadata_df.to_dict("records")
        elif mode == "test":
            # For test, we must tile the entire fragment dynamically
            for fid in unique_frag_ids:
                mask = self.fragments_data[fid]["mask"]
                h, w = mask.shape

                # Create tiles with stride = image_size (non-overlapping for inference efficiency)
                # Note: Overlapping inference can be better, but we start with standard tiling.
                for y in range(0, h, self.cfg.image_size):
                    for x in range(0, w, self.cfg.image_size):
                        self.samples.append(
                            {
                                "fragment_id": fid,
                                "x": x,
                                "y": y,
                                "width": self.cfg.image_size,
                                "height": self.cfg.image_size,
                            }
                        )

        self.transforms = get_transforms(mode)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fid = sample["fragment_id"]
        x = sample["x"]
        y = sample["y"]
        w = sample["width"]
        h = sample["height"]

        # Retrieve full fragment data
        data = self.fragments_data[fid]
        volume_full = data["volume"]  # Shape: (D, H_full, W_full)
        mask_full = data["mask"]
        label_full = data["label"]

        # --- 1. Determine Z-Jitter (Train Only) ---
        jitter = 0
        if self.mode == "train":
            # Random shift in [-z_jitter_range, +z_jitter_range]
            # e.g., if range is 2, jitter is in {-2, -1, 0, 1, 2}
            jitter = np.random.randint(
                -self.cfg.z_jitter_range, self.cfg.z_jitter_range + 1
            )

        # --- 2. Crop Volume ---
        vol_d, vol_h, vol_w = volume_full.shape

        # Calculate crop coordinates handling boundaries
        y_end = min(y + h, vol_h)
        x_end = min(x + w, vol_w)

        # Calculate padding if patch goes out of bounds
        pad_h = max(0, (y + h) - vol_h)
        pad_w = max(0, (x + w) - vol_w)

        # Extract crop from volume
        vol_crop = volume_full[:, y:y_end, x:x_end]

        # Pad spatial dimensions if necessary
        if pad_h > 0 or pad_w > 0:
            vol_crop = np.pad(
                vol_crop,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

        # --- 3. Compute 5-Channel Projection ---
        channels = []
        for start_z, end_z in self.cfg.z_ranges:
            # Apply jitter
            s = start_z + jitter
            e = end_z + jitter

            # Map global Z-index to local index in loaded volume
            # Loaded volume starts at LOAD_Z_START
            s_local = s - LOAD_Z_START
            e_local = e - LOAD_Z_START

            # Clamp to valid range of loaded volume
            s_local = max(0, min(s_local, vol_d))
            e_local = max(0, min(e_local, vol_d))

            if s_local >= e_local:
                # Empty slice range
                mip = np.zeros(
                    (self.cfg.image_size, self.cfg.image_size), dtype=vol_crop.dtype
                )
            else:
                # Extract slab and compute MIP
                slab = vol_crop[s_local:e_local, :, :]
                if slab.shape[0] > 0:
                    mip = np.max(slab, axis=0)
                else:
                    mip = np.zeros(
                        (self.cfg.image_size, self.cfg.image_size), dtype=vol_crop.dtype
                    )

            channels.append(mip)

        # Stack channels -> (5, H, W)
        image = np.stack(channels, axis=0)

        # Normalize to [0, 1] (Data is uint16)
        image = image.astype(np.float32) / 65535.0

        # Transpose to (H, W, 5) for Albumentations
        image = np.transpose(image, (1, 2, 0))

        # --- 4. Crop Label/Mask ---
        label_patch = None

        if label_full is not None:
            l_crop = label_full[y:y_end, x:x_end]
            if pad_h > 0 or pad_w > 0:
                l_crop = np.pad(
                    l_crop, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
                )
            label_patch = l_crop.astype(np.float32)
        else:
            # Dummy label for test set
            label_patch = np.zeros(
                (self.cfg.image_size, self.cfg.image_size), dtype=np.float32
            )

        # --- 5. Apply Augmentations ---
        if self.transforms:
            # Albumentations handles image and mask
            augmented = self.transforms(image=image, mask=label_patch)
            image = augmented["image"]  # Returns Tensor (C, H, W)
            label_patch = augmented["mask"]  # Returns Tensor (H, W)

        # Ensure label is (1, H, W)
        if label_patch.ndim == 2:
            label_patch = label_patch.unsqueeze(0)

        return image, label_patch, idx


def get_dataloaders(cfg):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    # Load Metadata
    train_df = pd.read_csv(cfg.train_metadata_path)
    val_df = pd.read_csv(cfg.val_metadata_path)
    test_df = pd.read_csv(cfg.test_metadata_path)

    # Initialize Datasets
    train_ds = InkDataset(train_df, mode="train", load_cached_data=True)
    val_ds = InkDataset(val_df, mode="validation", load_cached_data=True)
    test_ds = InkDataset(test_df, mode="test", load_cached_data=True)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
