import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Segmentation.

    Features:
    - Loads satellite bands 11, 14, 15.
    - Generates 6-channel input: 3-channel Ash Composite (t=4) + 3-channel Temporal Diff (t=4 - t=3).
    - Handles caching of processed tensors to optimize training runtime.
    - Applies affine-only augmentations during training.
    """

    def __init__(self, split="train", debug=False, load_cached_data=True):
        """
        Args:
            split (str): 'train', 'validation', or 'test'.
            debug (bool): If True, limits dataset to Config.DEBUG_SAMPLE_SIZE.
            load_cached_data (bool): If True, attempts to load/save processed data from disk.
        """
        self.split = split
        self.debug = debug
        self.load_cached_data = load_cached_data

        # 1. Setup Metadata Paths
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
            self.has_mask = True
        elif split == "validation":
            self.metadata_path = Config.VALID_METADATA_PATH
            self.has_mask = True
        elif split == "test":
            self.metadata_path = Config.TEST_METADATA_PATH
            self.has_mask = False
        else:
            raise ValueError(f"Unknown split: {split}")

        # 2. Load Metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

        # Debug Sampling
        if self.debug:
            self.df = self.df.sample(
                n=min(len(self.df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            ).reset_index(drop=True)

        # 3. Setup Caching
        # We cache the deterministic processed data (6-channel img + mask) before augmentation.
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache", split)
        if self.load_cached_data:
            os.makedirs(self.cache_dir, exist_ok=True)

        # 4. Setup Augmentations
        # We use ToTensorV2 with transpose_mask=True to ensure masks are (C, H, W)
        if split == "train":
            self.transform = A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=Config.AUG_SHIFT_LIMIT,
                        scale_limit=Config.AUG_SCALE_LIMIT,
                        rotate_limit=Config.AUG_ROTATION_LIMIT,
                        p=Config.AUG_PROB,
                        border_mode=0,  # cv2.BORDER_CONSTANT
                        value=0,
                        mask_value=0,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    ToTensorV2(transpose_mask=True),
                ]
            )
        else:
            self.transform = ToTensorV2(transpose_mask=True)

    def __len__(self):
        return len(self.df)

    def normalize(self, data, min_v, max_v):
        """
        Normalizes data to [0, 1] range based on physical min/max bounds.
        """
        return np.clip((data - min_v) / (max_v - min_v), 0, 1)

    def process_image(self, row):
        """
        Loads raw bands and computes the 6-channel input.
        Returns:
            img (np.ndarray): (H, W, 6) float32
            mask (np.ndarray): (H, W, 1) float32
        """
        # Load Bands 11, 14, 15
        bands_data = {}
        for band_idx in Config.ASH_BAND_IDS:
            band_col = f"band_{band_idx:02d}"
            path = os.path.join(Config.INPUT_DIR, row[band_col])

            try:
                # Load full sequence (H, W, T)
                full_seq = np.load(path)
                # Extract t=3 (prev) and t=4 (curr)
                bands_data[band_idx] = full_seq[
                    ..., [Config.PREV_FRAME_IDX, Config.LABELED_FRAME_IDX]
                ]
            except Exception as e:
                # Fallback (should not happen with valid data)
                bands_data[band_idx] = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 2), dtype=np.float32
                )

        b11 = bands_data[11]
        b14 = bands_data[14]
        b15 = bands_data[15]

        # --- Construct Ash Composite (t=4) ---
        # Index 1 corresponds to LABELED_FRAME_IDX (4)
        ash_curr = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.float32)
        ash_curr[..., 0] = self.normalize(
            b15[..., 1] - b14[..., 1], Config.ASH_MIN[0], Config.ASH_MAX[0]
        )
        ash_curr[..., 1] = self.normalize(
            b14[..., 1] - b11[..., 1], Config.ASH_MIN[1], Config.ASH_MAX[1]
        )
        ash_curr[..., 2] = self.normalize(
            b14[..., 1], Config.ASH_MIN[2], Config.ASH_MAX[2]
        )

        # --- Construct Ash Composite (t=3) ---
        # Index 0 corresponds to PREV_FRAME_IDX (3)
        ash_prev = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.float32)
        ash_prev[..., 0] = self.normalize(
            b15[..., 0] - b14[..., 0], Config.ASH_MIN[0], Config.ASH_MAX[0]
        )
        ash_prev[..., 1] = self.normalize(
            b14[..., 0] - b11[..., 0], Config.ASH_MIN[1], Config.ASH_MAX[1]
        )
        ash_prev[..., 2] = self.normalize(
            b14[..., 0], Config.ASH_MIN[2], Config.ASH_MAX[2]
        )

        # --- Temporal Difference ---
        ash_diff = ash_curr - ash_prev

        # --- Stack to 6 Channels ---
        img = np.concatenate([ash_curr, ash_diff], axis=-1)  # (H, W, 6)

        # --- Load Mask ---
        if self.has_mask:
            mask_path = os.path.join(Config.INPUT_DIR, row["human_pixel_masks"])
            mask = np.load(mask_path).astype(np.float32)  # (H, W, 1)
        else:
            mask = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1), dtype=np.float32)

        return img, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # Define cache path
        cache_file = os.path.join(self.cache_dir, f"{record_id}.npz")

        img = None
        mask = None

        # 1. Try Loading from Cache
        if self.load_cached_data and os.path.exists(cache_file):
            try:
                data = np.load(cache_file)
                img = data["img"]
                mask = data["mask"]
            except Exception:
                # Corrupt cache, recompute
                pass

        # 2. Process from Scratch if needed
        if img is None:
            img, mask = self.process_image(row)

            # Save to cache
            if self.load_cached_data:
                np.savez(cache_file, img=img, mask=mask)

        # 3. Apply Augmentations
        # Albumentations expects HWC inputs
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented["image"]  # (6, H, W)
            mask_tensor = augmented["mask"]  # (1, H, W) due to transpose_mask=True
        else:
            # Fallback manual conversion if transform is None (unlikely)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)

        return img_tensor, mask_tensor
