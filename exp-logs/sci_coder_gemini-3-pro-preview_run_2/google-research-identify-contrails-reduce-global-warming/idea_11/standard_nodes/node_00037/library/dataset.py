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
    - Loads satellite bands 11, 14, 15 from NPY files.
    - Generates a 6-channel input tensor:
        - Channels 1-3: "Ash" False Color Composite (t=4).
        - Channels 4-6: Temporal Difference (t=4 - t=3).
    - Applies Affine Augmentations (Rotation, Scale, Shift, Flip) for training.
    - Handles Train/Validation/Test splits via metadata CSVs.
    """

    def __init__(self, split="train", transform=None, debug=Config.DEBUG):
        """
        Args:
            split (str): 'train', 'validation', or 'test'.
            transform (albumentations.Compose): Custom transforms. If None, defaults are used.
            debug (bool): If True, subsets the dataset for debugging.
        """
        self.split = split
        self.debug = debug

        # Load appropriate metadata
        if split == "train":
            self.df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        elif split == "validation":
            self.df = pd.read_csv(Config.VALID_METADATA_PATH)
        elif split == "test":
            self.df = pd.read_csv(Config.TEST_METADATA_PATH)
        else:
            raise ValueError(f"Unknown split: {split}")

        # Debug Mode: Sample subset
        if self.debug:
            self.df = self.df.sample(
                n=min(len(self.df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            ).reset_index(drop=True)

        # Setup Transforms
        if transform is None:
            self.transform = self.get_default_transforms(split)
        else:
            self.transform = transform

    def get_default_transforms(self, split):
        """
        Returns the Albumentations transform pipeline.

        Train: Resize -> Affine (Flip, Shift, Scale, Rotate) -> ToTensor
        Valid/Test: Resize -> ToTensor
        """
        if split == "train":
            return A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    # Affine transformations only; avoid elastic/grid distortions for linear contrails
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=0.5,
                        border_mode=0,
                        value=0,
                        mask_value=0,
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose([A.Resize(Config.IMG_SIZE, Config.IMG_SIZE), ToTensorV2()])

    def normalize_range(self, data, min_val, max_val):
        """
        Normalizes data to [0, 1] based on provided min/max bounds.
        Clips values outside the range.
        """
        return np.clip((data - min_val) / (max_val - min_val), 0, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # ---------------------------------------------------------
        # 1. Load Satellite Bands
        # ---------------------------------------------------------
        # We need Bands 11, 14, 15 for the Ash composite and temporal diffs.
        # Files are (H, W, T) where T=8.
        # Index 4 is the labeled frame (t=0 relative to label).
        # Index 3 is the previous frame (t=-10 min).

        try:
            p11 = os.path.join(Config.INPUT_DIR, row["band_11"])
            p14 = os.path.join(Config.INPUT_DIR, row["band_14"])
            p15 = os.path.join(Config.INPUT_DIR, row["band_15"])

            b11 = np.load(p11)
            b14 = np.load(p14)
            b15 = np.load(p15)

            # Extract time steps
            idx_t4 = 4  # Current
            idx_t3 = 3  # Previous

            t11_t4, t11_t3 = b11[..., idx_t4], b11[..., idx_t3]
            t14_t4, t14_t3 = b14[..., idx_t4], b14[..., idx_t3]
            t15_t4, t15_t3 = b15[..., idx_t4], b15[..., idx_t3]

        except Exception as e:
            # Fallback for corrupt/missing files (unlikely given EDA)
            print(f"Error loading record {record_id}: {e}")
            img = torch.zeros(
                (Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE),
                dtype=torch.float32,
            )
            if self.split == "test":
                return img, record_id
            else:
                return img, torch.zeros(
                    (1, Config.IMG_SIZE, Config.IMG_SIZE), dtype=torch.float32
                )

        # ---------------------------------------------------------
        # 2. Feature Engineering (6 Channels)
        # ---------------------------------------------------------

        # --- Channels 1-3: "Ash" False Color Composite (t=4) ---
        # Red: T15 - T14 (Optical Depth proxy)
        # Green: T14 - T11 (Particle Phase proxy)
        # Blue: T14 (Temperature proxy)
        # Bounds derived from standard Ash RGB recipes.
        r_ash = self.normalize_range(t15_t4 - t14_t4, -4, 2)
        g_ash = self.normalize_range(t14_t4 - t11_t4, -4, 5)
        b_ash = self.normalize_range(t14_t4, 243, 303)

        # --- Channels 4-6: Temporal Difference (t=4 - t=3) ---
        # Captures motion and evolution.
        # Normalized assuming a range of [-2K, 2K] for 10-min intervals.
        r_diff = self.normalize_range(t15_t4 - t15_t3, -2, 2)
        g_diff = self.normalize_range(t14_t4 - t14_t3, -2, 2)
        b_diff = self.normalize_range(t11_t4 - t11_t3, -2, 2)

        # Stack to create (H, W, 6) image
        img = np.stack([r_ash, g_ash, b_ash, r_diff, g_diff, b_diff], axis=-1).astype(
            np.float32
        )

        # ---------------------------------------------------------
        # 3. Load Mask (Train/Validation only)
        # ---------------------------------------------------------
        mask = None
        if self.split != "test":
            mask_path = os.path.join(Config.INPUT_DIR, row["human_pixel_masks"])
            mask = np.load(mask_path).astype(np.float32)  # (H, W, 1)

        # ---------------------------------------------------------
        # 4. Apply Augmentations
        # ---------------------------------------------------------
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]  # (6, H, W) tensor
                mask = augmented["mask"]  # (H, W, 1) tensor (usually)

                # Ensure mask is (C, H, W) -> (1, H, W)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.ndim == 3 and mask.shape[-1] == 1:
                    mask = mask.permute(2, 0, 1)
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]

        # ---------------------------------------------------------
        # 5. Return
        # ---------------------------------------------------------
        if self.split == "test":
            return img, record_id
        else:
            return img, mask
