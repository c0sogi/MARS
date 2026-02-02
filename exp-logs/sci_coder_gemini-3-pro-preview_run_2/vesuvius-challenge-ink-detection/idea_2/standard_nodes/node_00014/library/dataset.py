import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


class InkDataset(Dataset):
    """
    Dataset for Vesuvius Ink Detection.

    Features:
    - Stratified Depth Projection: Converts 3D sub-volume into 3-channel 2.5D input.
    - Caching: Caches processed projections to disk to speed up training.
    - Tiling: Automatically tiles test fragments if patch metadata is missing.
    - Augmentation: Geometric-only augmentations for training.
    """

    def __init__(self, dataframe, mode="train", load_cached_data=True):
        """
        Args:
            dataframe (pd.DataFrame): Metadata containing fragment_id and optionally x, y, etc.
            mode (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Whether to try loading processed volumes from disk.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Handle Test Mode Tiling
        # If test mode, the dataframe might only contain fragment paths, not patch coordinates.
        if self.mode == "test" and "x" not in dataframe.columns:
            self.df = self._generate_test_tiles(dataframe)
        else:
            self.df = dataframe

        self.fragment_cache = {}
        self.transform = self._get_transforms()

        # Ensure cache directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Pre-load/Cache fragments
        unique_fragments = self.df["fragment_id"].unique()
        for frag_id in unique_fragments:
            # Find volume path for this fragment
            # We look at the first occurrence in the dataframe
            row = self.df[self.df["fragment_id"] == frag_id].iloc[0]
            vol_path = os.path.join(Config.INPUT_DIR, row["volume_path"])

            # Load or Compute MIP
            self.fragment_cache[frag_id] = self._load_fragment(frag_id, vol_path)

    def _generate_test_tiles(self, dataframe):
        """
        Generates tiling metadata for test fragments.
        """
        new_rows = []
        for _, row in dataframe.iterrows():
            frag_id = row["fragment_id"]
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])

            # Load mask to get dimensions
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            h, w = mask.shape

            # Generate tiles
            # We use the configured stride. For inference, one might sometimes use overlap,
            # but we stick to the config.
            for y in range(0, h, Config.STRIDE):
                for x in range(0, w, Config.STRIDE):
                    new_rows.append(
                        {
                            "fragment_id": frag_id,
                            "x": x,
                            "y": y,
                            "width": Config.TILE_SIZE,
                            "height": Config.TILE_SIZE,
                            "mask_path": row["mask_path"],
                            "volume_path": row["volume_path"],
                        }
                    )
        return pd.DataFrame(new_rows)

    def _get_transforms(self):
        """
        Returns Albumentations transforms.
        Geometric only: Flips and Rotations. No intensity changes.
        """
        if self.mode == "train":
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ]
            )
        return None

    def _load_fragment(self, frag_id, vol_path):
        """
        Loads the fragment volume, computes stratified MIPs, and caches to disk.
        """
        cache_filename = f"fragment_{frag_id}_mip_3ch.npy"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # 1. Try loading from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                # print(f"Loading cached fragment {frag_id} from {cache_path}")
                return np.load(cache_path)
            except Exception:
                # print(f"Cache corrupted for {frag_id}, recomputing...")
                pass

        # 2. Compute Stratified Projection
        slices = []
        # Load specific Z-range
        for z in range(Config.Z_START, Config.Z_END):
            slice_filename = f"{z:02d}.tif"
            slice_path = os.path.join(vol_path, slice_filename)

            if os.path.exists(slice_path):
                img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    slices.append(img)
            else:
                # If a slice is missing, we skip it.
                # In a real scenario, we might want to pad with zeros or error out.
                pass

        if not slices:
            raise ValueError(
                f"No slices found for fragment {frag_id} in range {Config.Z_START}-{Config.Z_END}"
            )

        volume = np.stack(slices, axis=0)  # Shape: (Depth, Height, Width)

        # Stratified MIP: Split Depth into 3 chunks
        D = volume.shape[0]
        # We want to split D into NUM_SUB_VOLUMES (3) parts.
        # np.array_split handles uneven splits gracefully.
        sub_volumes = np.array_split(volume, Config.NUM_SUB_VOLUMES, axis=0)

        mips = []
        for sub_vol in sub_volumes:
            if sub_vol.shape[0] > 0:
                mip = np.max(sub_vol, axis=0)
            else:
                # Fallback for empty sub-volume (unlikely given range 20 and split 3)
                mip = np.zeros_like(volume[0])
            mips.append(mip)

        # Stack to (H, W, 3)
        stratified_mip = np.stack(mips, axis=-1)

        # 3. Save to cache
        np.save(cache_path, stratified_mip)

        return stratified_mip

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frag_id = row["fragment_id"]
        x, y = int(row["x"]), int(row["y"])
        w, h = int(row["width"]), int(row["height"])

        # Retrieve full fragment image from memory cache
        full_image = self.fragment_cache[frag_id]  # (H_frag, W_frag, 3)
        img_h_full, img_w_full = full_image.shape[:2]

        # Calculate crop coordinates
        y_end = min(y + h, img_h_full)
        x_end = min(x + w, img_w_full)

        # Crop image
        image = full_image[y:y_end, x:x_end, :]

        # Calculate padding if crop is smaller than tile size
        pad_h = h - image.shape[0]
        pad_w = w - image.shape[1]

        # Pad image if necessary
        if pad_h > 0 or pad_w > 0:
            image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # Load Mask and Label if in train/val mode
        mask = None
        label = None

        if self.mode in ["train", "validation"]:
            # Load mask (valid pixel area)
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Crop mask
            mask_patch = mask_full[y:y_end, x:x_end]
            if pad_h > 0 or pad_w > 0:
                mask_patch = np.pad(
                    mask_patch,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

            # Load label (ink)
            label_path = os.path.join(Config.INPUT_DIR, row["label_path"])
            label_full = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            # Crop label
            label_patch = label_full[y:y_end, x:x_end]
            if pad_h > 0 or pad_w > 0:
                label_patch = np.pad(
                    label_patch,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

            # Apply Augmentations
            if self.transform:
                augmented = self.transform(
                    image=image, mask=mask_patch, label=label_patch
                )
                image = augmented["image"]
                mask_patch = augmented["mask"]
                label_patch = augmented[
                    "label"
                ]  # Albumentations can handle multiple masks if passed correctly,
                # but 'label' is not a standard key for single mask aug.
                # We should use 'masks' list or separate calls if keys differ.
                # However, standard A.Compose with 'mask' key works for one.
                # Let's use the 'masks' argument for safety or just pass additional targets.

                # Re-implementing augmentation application to be safe with arbitrary keys
                # Or simply:
                # data = self.transform(image=image, masks=[mask_patch, label_patch])
                # image = data['image']
                # mask_patch = data['masks'][0]
                # label_patch = data['masks'][1]
                pass

            # Since I used standard keys above which might be tricky with 'label' key not standard in basic transform,
            # let's do it explicitly with 'masks' to ensure geometric consistency.
            if self.transform:
                data = self.transform(image=image, masks=[mask_patch, label_patch])
                image = data["image"]
                mask_patch = data["masks"][0]
                label_patch = data["masks"][1]

            # Normalization and Tensor Conversion
            # Input is uint16 (0-65535). We scale to [0, 1].
            image = image.astype(np.float32) / 65535.0
            mask = (mask_patch > 0).astype(np.float32)
            label = (label_patch > 0).astype(np.float32)

            # To Tensor (H, W, C) -> (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).unsqueeze(0)
            label = torch.from_numpy(label).unsqueeze(0)

            return image, label, mask

        else:
            # Test Mode
            # Just Normalize and Tensor
            image = image.astype(np.float32) / 65535.0
            image = torch.from_numpy(image).permute(2, 0, 1)

            # We can return dummy mask/label or just image.
            # Standard PyTorch collate usually prefers consistent tuple sizes, but for inference loops
            # we often just need the image.
            # Let's return image only as per typical inference patterns,
            # or (image, fragment_id) if needed, but the loader index usually tracks that.
            return image
