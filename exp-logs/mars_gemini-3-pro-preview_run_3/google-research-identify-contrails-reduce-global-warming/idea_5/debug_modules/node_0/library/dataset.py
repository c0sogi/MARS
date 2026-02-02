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
    Dataset class for Contrail Detection.
    Constructs a 6-channel input tensor based on the Ash false-color composite
    and its temporal difference.
    """

    def __init__(self, split="train", transform=None, debug=None):
        """
        Args:
            split (str): One of 'train', 'validation', 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
            debug (bool): If True, uses a small subset of data for debugging.
        """
        self.split = split
        self.transform = transform
        self.debug = debug if debug is not None else Config.DEBUG

        # Load Metadata
        if split == "train":
            self.df = pd.read_csv(Config.TRAIN_METADATA)
        elif split == "validation":
            self.df = pd.read_csv(Config.VALIDATION_METADATA)
        elif split == "test":
            self.df = pd.read_csv(Config.TEST_METADATA)
        else:
            raise ValueError(f"Unknown split: {split}")

        # Handle Debugging
        if self.debug:
            self.df = self.df.sample(
                n=min(len(self.df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            ).reset_index(drop=True)

        # Ash Composite Normalization Bounds
        # Channels: R=(T15-T13), G=(T14-T11), B=T13
        self.ash_min = np.array([-4, -4, 243], dtype=np.float32)
        self.ash_max = np.array([2, 5, 303], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def normalize(self, img):
        """
        Normalizes Ash composite channels to [0, 1].
        img: (H, W, 3)
        """
        return np.clip((img - self.ash_min) / (self.ash_max - self.ash_min), 0, 1)

    def load_band(self, path):
        """
        Loads a single band from disk.
        path: Relative path to the .npy file.
        """
        full_path = os.path.join(Config.INPUT_DIR, path)
        try:
            return np.load(full_path)
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            # Return zeros as fallback
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 9), dtype=np.float32)

    def get_ash(self, b11, b13, b14, b15):
        """
        Computes Ash false-color composite.
        Args:
            b11, b13, b14, b15: 2D arrays of brightness temperatures.
        Returns:
            (H, W, 3) array.
        """
        r = b15 - b13
        g = b14 - b11
        b = b13
        return np.stack([r, g, b], axis=-1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Bands
        # We need the temporal sequence for bands 11, 13, 14, 15
        b11_seq = self.load_band(row["band_11"])
        b13_seq = self.load_band(row["band_13"])
        b14_seq = self.load_band(row["band_14"])
        b15_seq = self.load_band(row["band_15"])

        # 2. Extract Temporal Frames
        # t=4 is the labeled frame, t=3 is the previous frame
        t_idx = 4
        prev_idx = 3

        # Helper to safely extract frame
        def get_frame(seq, t):
            if seq.ndim == 3 and seq.shape[2] > t:
                return seq[:, :, t]
            # Fallback
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        b11_t = get_frame(b11_seq, t_idx)
        b13_t = get_frame(b13_seq, t_idx)
        b14_t = get_frame(b14_seq, t_idx)
        b15_t = get_frame(b15_seq, t_idx)

        b11_prev = get_frame(b11_seq, prev_idx)
        b13_prev = get_frame(b13_seq, prev_idx)
        b14_prev = get_frame(b14_seq, prev_idx)
        b15_prev = get_frame(b15_seq, prev_idx)

        # 3. Compute Ash Composites
        ash_t = self.get_ash(b11_t, b13_t, b14_t, b15_t)
        ash_prev = self.get_ash(b11_prev, b13_prev, b14_prev, b15_prev)

        # 4. Normalize
        ash_t = self.normalize(ash_t)
        ash_prev = self.normalize(ash_prev)

        # 5. Construct 6-Channel Input
        # Channels 0-2: Ash at t
        # Channels 3-5: Ash difference (t - prev)
        diff = ash_t - ash_prev
        image = np.concatenate([ash_t, diff], axis=-1)  # (H, W, 6)

        # 6. Load Mask (if available)
        mask = None
        if self.split in ["train", "validation"]:
            mask_path = os.path.join(Config.INPUT_DIR, row["human_pixel_masks"])
            if os.path.exists(mask_path):
                mask = np.load(mask_path)  # (H, W, 1)
            else:
                mask = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 1), dtype=np.float32)

        # 7. Apply Augmentations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]
        else:
            # Manual ToTensor if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            if mask is not None:
                mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()

        # 8. Return
        if self.split in ["train", "validation"]:
            return image, mask
        else:
            # For test set, return record_id for submission
            return image, str(row["record_id"])


def get_transforms(split="train"):
    """
    Returns the Albumentations transform pipeline.
    """
    if split == "train":
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])
