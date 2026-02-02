import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


class InkDataset(Dataset):
    """
    PyTorch Dataset for Vesuvius Ink Detection.

    Handles:
    1. Loading 3D sub-volumes from disk (caching to .npy for speed).
    2. On-the-fly Maximum Intensity Projection (MIP) generation.
    3. Volumetric Z-Jitter augmentation during training.
    4. Z-Scanning offset support for inference.
    5. Geometric augmentations and normalization.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        mode: str = "train",
        z_offset: int = 0,
        load_cached_data: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            metadata_df (pd.DataFrame): Dataframe containing patch metadata (fragment_id, x, y, etc.).
            mode (str): 'train', 'validation', or 'test'. Controls augmentation and jitter.
            z_offset (int): Shift for the Z-window (used for inference Z-scanning).
            load_cached_data (bool): If True, attempts to load pre-processed volumes from disk cache.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.mode = mode
        self.z_offset = z_offset
        self.load_cached_data = load_cached_data

        # Define the range of slices needed to cover all potential jitters and channels
        # Min Z: Z_START - Z_JITTER
        # Max Z: Z_START + Z_JITTER + (Total Depth of 3 channels)
        # Channels: [start, start+12], [start+6, start+18], [start+12, start+24]
        # Max relative depth is 24.
        self.min_slice = Config.Z_START - Config.Z_JITTER
        self.max_slice = Config.Z_START + Config.Z_JITTER + 24

        # Ensure slice bounds are non-negative
        if self.min_slice < 0:
            self.min_slice = 0

        # Cache for loaded fragment volumes (in memory)
        # Structure: {fragment_id: {'volume': np.ndarray, 'mask': np.ndarray, 'label': np.ndarray}}
        self.fragments = {}

        # Pre-load data
        self._preload_fragments()

        # Define Augmentations
        if self.mode == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    ToTensorV2(),
                ],
                additional_targets={"valid_mask": "mask", "ink_label": "mask"},
            )
        else:
            self.transform = A.Compose(
                [
                    ToTensorV2(),
                ],
                additional_targets={"valid_mask": "mask", "ink_label": "mask"},
            )

    def _preload_fragments(self):
        """
        Loads necessary data for all fragments present in the metadata into memory.
        Implements disk caching mechanism.
        """
        unique_fragments = self.metadata["fragment_id"].unique()

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        for frag_id in unique_fragments:
            # Get paths from the first entry for this fragment
            frag_meta = self.metadata[self.metadata["fragment_id"] == frag_id].iloc[0]

            # 1. Load Volume
            # We load a "slab" of the volume sufficient for all Z-jitters
            volume = self._load_volume_slab(frag_id, frag_meta["volume_path"])

            # 2. Load Binary Mask (Valid Pixels)
            mask_path = os.path.join(Config.INPUT_DIR, frag_meta["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Fallback for test set if mask structure differs or file missing (should not happen based on spec)
                raise FileNotFoundError(f"Mask not found at {mask_path}")
            mask = (mask > 0).astype(np.uint8)

            # 3. Load Ink Label (Ground Truth) - Only for train/val
            label = None
            if self.mode in ["train", "validation"]:
                label_path = os.path.join(Config.INPUT_DIR, frag_meta["label_path"])
                if os.path.exists(label_path):
                    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    label = (label_img > 0).astype(np.uint8)
                else:
                    # Fallback if label missing (should not happen for train/val)
                    label = np.zeros_like(mask)

            self.fragments[frag_id] = {
                "volume": volume,  # Shape: (D, H, W)
                "mask": mask,  # Shape: (H, W)
                "label": label,  # Shape: (H, W) or None
            }

    def _load_volume_slab(self, frag_id, rel_vol_path):
        """
        Loads the specific Z-slab of the volume.
        Uses caching: checks .npy first, else loads from .tif slices and saves .npy.
        """
        cache_path = os.path.join(
            Config.CACHE_DIR,
            f"frag_{frag_id}_slab_{self.min_slice}_{self.max_slice}.npy",
        )

        if self.load_cached_data and os.path.exists(cache_path):
            try:
                # Load from cache
                volume = np.load(cache_path)
                return volume
            except Exception as e:
                print(
                    f"Failed to load cache for {frag_id}: {e}. Reloading from source."
                )

        # Load from source TIFFs
        vol_dir = os.path.join(Config.INPUT_DIR, rel_vol_path)
        slices = []

        # We assume filenames are sorted and correspond to indices 00.tif, 01.tif, etc.
        # We only load the range [self.min_slice, self.max_slice)
        for z in range(self.min_slice, self.max_slice):
            filename = f"{z:02d}.tif"
            file_path = os.path.join(vol_dir, filename)

            if not os.path.exists(file_path):
                # If slice is out of bounds (e.g. fragment has fewer slices), pad with zeros
                # Get shape from a previous slice or mask
                # Assuming at least one slice exists or we can infer shape from mask
                # For simplicity, we assume data integrity based on competition spec
                print(
                    f"Warning: Slice {filename} missing for {frag_id}. Padding with zeros."
                )
                # Need shape. We can get it from mask path in metadata if needed,
                # but here we defer until we have a valid slice or use the mask loaded later.
                # Just append None and fix later or raise error.
                raise FileNotFoundError(f"Slice {file_path} not found.")

            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            slices.append(img)

        volume = np.stack(slices, axis=0)  # (D, H, W)

        # Save to cache
        np.save(cache_path, volume)

        return volume

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        frag_id = row["fragment_id"]
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        # Retrieve full fragment data
        frag_data = self.fragments[frag_id]
        full_volume = frag_data["volume"]
        full_mask = frag_data["mask"]
        full_label = frag_data["label"]

        # --- Determine Z-Start Index ---
        if self.mode == "train":
            # Volumetric Z-Jitter
            jitter = np.random.randint(-Config.Z_JITTER, Config.Z_JITTER + 1)
            current_z_start = Config.Z_START + jitter
        else:
            # Deterministic for Val/Test (plus optional inference offset)
            current_z_start = Config.Z_START + self.z_offset

        # --- Extract Channels (MIPs) ---
        # Calculate indices relative to the loaded volume slab
        # The loaded volume starts at self.min_slice.
        # So index 0 in full_volume corresponds to self.min_slice in absolute Z.
        rel_start = current_z_start - self.min_slice

        # Get channel ranges from Config
        # Config.get_channel_indices returns absolute indices [(s1, e1), ...]
        # We need to convert them to relative indices for our loaded slab
        abs_ranges = Config.get_channel_indices(current_z_start)

        channels = []
        for abs_s, abs_e in abs_ranges:
            # Convert to relative indices
            rel_s = abs_s - self.min_slice
            rel_e = abs_e - self.min_slice

            # Boundary checks
            rel_s = max(0, rel_s)
            rel_e = min(full_volume.shape[0], rel_e)

            if rel_s >= rel_e:
                # Fallback if range is invalid (should not happen with correct padding)
                mip = np.zeros((h, w), dtype=np.float32)
            else:
                # Extract patch from volume first to save compute
                # Volume shape: (D, H, W)
                # We need volume[rel_s:rel_e, y:y+h, x:x+w]
                # Handle edge cases for x, y (if patch goes out of bounds)
                # The metadata generator ensures validity, but we clip to be safe.
                img_h, img_w = full_volume.shape[1], full_volume.shape[2]
                y_end = min(y + h, img_h)
                x_end = min(x + w, img_w)

                vol_crop = full_volume[rel_s:rel_e, y:y_end, x:x_end]

                # Compute MIP (Maximum Intensity Projection) along Z-axis
                if vol_crop.size == 0:
                    mip = np.zeros((h, w), dtype=full_volume.dtype)
                else:
                    mip = np.max(vol_crop, axis=0)

                # Pad if crop was smaller than tile size
                if mip.shape[0] < h or mip.shape[1] < w:
                    mip = np.pad(
                        mip,
                        ((0, h - mip.shape[0]), (0, w - mip.shape[1])),
                        mode="constant",
                    )

            channels.append(mip)

        # Stack channels: (H, W, 3)
        image = np.stack(channels, axis=-1)

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 65535.0

        # --- Extract Mask and Label ---
        img_h, img_w = full_mask.shape
        y_end = min(y + h, img_h)
        x_end = min(x + w, img_w)

        mask_crop = full_mask[y:y_end, x:x_end]
        if mask_crop.shape[0] < h or mask_crop.shape[1] < w:
            mask_crop = np.pad(
                mask_crop,
                ((0, h - mask_crop.shape[0]), (0, w - mask_crop.shape[1])),
                mode="constant",
            )

        if full_label is not None:
            label_crop = full_label[y:y_end, x:x_end]
            if label_crop.shape[0] < h or label_crop.shape[1] < w:
                label_crop = np.pad(
                    label_crop,
                    ((0, h - label_crop.shape[0]), (0, w - label_crop.shape[1])),
                    mode="constant",
                )
        else:
            label_crop = np.zeros((h, w), dtype=np.uint8)

        # --- Augmentation ---
        # Albumentations expects HWC for image
        augmented = self.transform(
            image=image, valid_mask=mask_crop, ink_label=label_crop
        )

        image_tensor = augmented["image"]  # (3, H, W)
        mask_tensor = augmented[
            "valid_mask"
        ]  # (H, W) -> will become (1, H, W) via unsqueeze later if needed
        label_tensor = augmented["ink_label"]  # (H, W)

        # Add channel dim to mask and label for consistency
        mask_tensor = mask_tensor.unsqueeze(0).float()
        label_tensor = label_tensor.unsqueeze(0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,  # The valid pixel mask
            "label": label_tensor,  # The ink label
            "fragment_id": frag_id,
            "coords": torch.tensor([x, y]),
        }
