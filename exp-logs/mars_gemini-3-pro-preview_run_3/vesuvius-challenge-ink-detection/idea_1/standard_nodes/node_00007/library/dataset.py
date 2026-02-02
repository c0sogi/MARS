import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List

# Import configuration and utilities
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    CACHE_DIR,
    Z_DIM,
    PATCH_SIZE,
    seed_everything,
)


class InkDataset(Dataset):
    """
    Dataset class for 3D Ink Detection.
    Loads 3D surface volumes and corresponding masks/labels.
    Provides random crops for training/validation.
    """

    def __init__(
        self,
        split: str,
        patches_per_epoch: int = 1000,
        load_cached_data: bool = True,
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            patches_per_epoch (int): Number of patches to generate per epoch.
            load_cached_data (bool): Whether to load data from cache if available.
        """
        self.split = split
        self.patches_per_epoch = patches_per_epoch
        self.load_cached_data = load_cached_data

        # Metadata path
        self.meta_path = METADATA_DIR / f"{split}.csv"
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        self.meta_df = pd.read_csv(self.meta_path)

        # Storage for loaded data
        self.fragments = []
        self.fragment_ids = []

        # Load all fragments into memory
        self._load_all_fragments()

        # Compute or load normalization stats
        self._normalize_data()

    def _load_all_fragments(self):
        """Iterates through metadata and loads each fragment."""
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        for _, row in self.meta_df.iterrows():
            frag_id = str(row["fragment_id"])

            # Define cache paths
            cache_vol_path = CACHE_DIR / f"{self.split}_{frag_id}_volume.npy"
            cache_lbl_path = CACHE_DIR / f"{self.split}_{frag_id}_label.npy"
            cache_msk_path = CACHE_DIR / f"{self.split}_{frag_id}_mask.npy"

            volume = None
            label = None
            mask = None

            # 1. Try Loading from Cache
            if self.load_cached_data:
                if (
                    cache_vol_path.exists()
                    and cache_msk_path.exists()
                    and (self.split == "test" or cache_lbl_path.exists())
                ):

                    try:
                        volume = np.load(cache_vol_path)
                        mask = np.load(cache_msk_path)
                        if self.split != "test":
                            label = np.load(cache_lbl_path)
                    except Exception as e:
                        print(
                            f"Failed to load cache for {frag_id}: {e}. Reloading raw."
                        )
                        volume = None  # Trigger raw load

            # 2. Load Raw Data if needed
            if volume is None:
                # Load Volume (65 slices)
                vol_path = INPUT_DIR / row["surface_volume_path"]
                slices = []
                for i in range(Z_DIM):
                    slice_path = vol_path / f"{i:02d}.tif"
                    if not slice_path.exists():
                        raise FileNotFoundError(f"Slice missing: {slice_path}")
                    # Load as grayscale
                    img = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
                    slices.append(img)

                # Stack to (Z, H, W)
                volume = np.stack(slices, axis=0)

                # Load Mask
                mask_path = INPUT_DIR / row["mask_path"]
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = (mask > 0).astype(np.uint8)  # Binary 0/1

                # Load Label (if not test)
                if self.split != "test":
                    lbl_path = INPUT_DIR / row["inklabels_path"]
                    label = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
                    label = (label > 0).astype(np.uint8)  # Binary 0/1

                # Save to Cache
                np.save(cache_vol_path, volume)
                np.save(cache_msk_path, mask)
                if self.split != "test":
                    np.save(cache_lbl_path, label)

            # Pre-calculate valid indices for faster sampling
            # We erode the mask slightly to avoid sampling edges where crop might go out of bounds
            # But simpler is to just sample from mask and handle bounds in __getitem__
            valid_indices = np.argwhere(mask > 0)

            self.fragments.append(
                {
                    "id": frag_id,
                    "volume": volume,  # Shape: (Z, H, W)
                    "label": label,  # Shape: (H, W) or None
                    "mask": mask,  # Shape: (H, W)
                    "valid_indices": valid_indices,  # Shape: (N, 2) -> (y, x)
                }
            )
            self.fragment_ids.append(frag_id)

    def _normalize_data(self):
        """Computes global mean/std and normalizes volumes in-place."""
        # Check if stats are cached
        stats_path = CACHE_DIR / "normalization_stats.npy"

        if self.load_cached_data and stats_path.exists():
            stats = np.load(stats_path, allow_pickle=True).item()
            mean = stats["mean"]
            std = stats["std"]
        else:
            # Compute stats from loaded fragments
            # To save time, we can subsample pixels
            print(f"Computing normalization stats for {self.split}...")
            pixel_sum = 0.0
            pixel_sq_sum = 0.0
            pixel_count = 0

            for frag in self.fragments:
                vol = frag["volume"].astype(np.float32)
                # Subsample for speed (e.g., every 100th pixel)
                subsample = vol[:, ::10, ::10]
                pixel_sum += np.sum(subsample)
                pixel_sq_sum += np.sum(subsample**2)
                pixel_count += subsample.size

            mean = pixel_sum / pixel_count
            std = np.sqrt((pixel_sq_sum / pixel_count) - (mean**2))

            # Save stats
            np.save(stats_path, {"mean": mean, "std": std})
            print(f"Stats computed: Mean={mean:.4f}, Std={std:.4f}")

        # Apply normalization
        for frag in self.fragments:
            # Convert to float32 and normalize
            frag["volume"] = (frag["volume"].astype(np.float32) - mean) / (std + 1e-6)

    def __len__(self):
        return self.patches_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a random crop from a random fragment.

        Returns:
            volume_patch: (Z, PATCH_SIZE, PATCH_SIZE)
            label_patch: (1, PATCH_SIZE, PATCH_SIZE)
        """
        # 1. Select random fragment
        frag_idx = np.random.randint(len(self.fragments))
        frag = self.fragments[frag_idx]

        # 2. Select random center from valid mask indices
        # valid_indices contains (y, x) coordinates where mask is 1
        valid_indices = frag["valid_indices"]
        if len(valid_indices) == 0:
            # Fallback for empty mask (unlikely)
            center_y, center_x = (
                frag["volume"].shape[1] // 2,
                frag["volume"].shape[2] // 2,
            )
        else:
            rand_pt_idx = np.random.randint(len(valid_indices))
            center_y, center_x = valid_indices[rand_pt_idx]

        # 3. Calculate crop coordinates
        half_size = PATCH_SIZE // 2
        y1 = center_y - half_size
        x1 = center_x - half_size
        y2 = y1 + PATCH_SIZE
        x2 = x1 + PATCH_SIZE

        # 4. Handle Boundaries (Shift window to fit)
        H, W = frag["volume"].shape[1], frag["volume"].shape[2]

        if y1 < 0:
            y1, y2 = 0, PATCH_SIZE
        elif y2 > H:
            y1, y2 = H - PATCH_SIZE, H

        if x1 < 0:
            x1, x2 = 0, PATCH_SIZE
        elif x2 > W:
            x1, x2 = W - PATCH_SIZE, W

        # 5. Crop
        # Volume: (Z, H, W) -> (Z, Patch, Patch)
        vol_crop = frag["volume"][:, y1:y2, x1:x2]

        # Label: (H, W) -> (1, Patch, Patch)
        if frag["label"] is not None:
            lbl_crop = frag["label"][y1:y2, x1:x2]
            lbl_crop = lbl_crop[np.newaxis, :, :]  # Add channel dim
        else:
            # For test set, return dummy label
            lbl_crop = np.zeros((1, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

        # 6. Geometric Augmentations
        # Cite solution_lesson_node_00005: Apply random rotations and flips for rotational invariance.
        if self.split == "train":
            # Random rotation (0, 90, 180, 270 degrees)
            k = np.random.randint(0, 4)
            vol_crop = np.rot90(vol_crop, k=k, axes=(1, 2))
            lbl_crop = np.rot90(lbl_crop, k=k, axes=(1, 2))

            # Random Horizontal Flip
            if np.random.rand() < 0.5:
                vol_crop = np.flip(vol_crop, axis=2)
                lbl_crop = np.flip(lbl_crop, axis=2)

            # Random Vertical Flip
            if np.random.rand() < 0.5:
                vol_crop = np.flip(vol_crop, axis=1)
                lbl_crop = np.flip(lbl_crop, axis=1)

        # Ensure arrays are contiguous (required for torch.from_numpy after flip/rot)
        vol_crop = np.ascontiguousarray(vol_crop)
        lbl_crop = np.ascontiguousarray(lbl_crop)

        # Convert to tensors
        vol_tensor = torch.from_numpy(vol_crop).float()
        lbl_tensor = torch.from_numpy(lbl_crop).float()

        return vol_tensor, lbl_tensor
