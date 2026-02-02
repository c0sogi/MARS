import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import CFG


def get_transforms(split="train"):
    """
    Returns the Albumentations transform pipeline for the specified split.

    Args:
        split (str): 'train', 'validation', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Strict Affine transformations as per Idea 15 (Rotation, Scale, Shift)
                # Elastic/Grid distortions are explicitly excluded to preserve linear morphology.
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=0,  # Constant padding with 0
                    value=0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Just convert to tensor
        return A.Compose([ToTensorV2()])


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Identification.
    Handles loading, preprocessing (Ash scheme), caching, and augmentation.
    """

    def __init__(
        self,
        metadata_path,
        split="train",
        transform=None,
        debug=False,
        debug_sample_size=None,
        load_cached_data=True,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            split (str): 'train', 'validation', or 'test'.
            transform (A.Compose): Albumentations transforms.
            debug (bool): If True, limits dataset size.
            debug_sample_size (int): Number of samples to use in debug mode.
            load_cached_data (bool): Whether to use disk caching for processed inputs.
        """
        self.split = split
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Load Metadata
        self.df = pd.read_csv(metadata_path)
        # Ensure record_id is string
        self.df["record_id"] = self.df["record_id"].astype(str)

        # Debug Mode
        if debug:
            sample_size = (
                debug_sample_size
                if debug_sample_size is not None
                else CFG.debug_sample_size
            )
            self.df = self.df.sample(
                n=min(len(self.df), sample_size), random_state=CFG.seed
            ).reset_index(drop=True)

        # Setup Cache Directory
        self.cache_dir = os.path.join(CFG.working_dir, "cache", self.split)
        if self.load_cached_data:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Ash Color Scheme Bounds
        # (min, max)
        self.bounds = {
            "T15_T14": (-6.7, 2.6),  # Red channel component
            "T14_T11": (-6.0, 6.3),  # Green channel component
            "T14": (243, 303),  # Blue channel component
        }

    def __len__(self):
        return len(self.df)

    def normalize(self, data, min_val, max_val):
        """Linearly normalizes data to [0, 1] based on provided bounds."""
        return (data - min_val) / (max_val - min_val)

    def get_ash_composite(self, b11, b14, b15):
        """
        Computes the 'Ash' False Color Composite.

        Args:
            b11, b14, b15: Numpy arrays of shape (H, W) or (H, W, T)

        Returns:
            np.ndarray: Ash composite normalized to [0, 1].
        """
        # Red: T15 - T14
        r = self.normalize(b15 - b14, *self.bounds["T15_T14"])

        # Green: T14 - T11
        g = self.normalize(b14 - b11, *self.bounds["T14_T11"])

        # Blue: T14
        b = self.normalize(b14, *self.bounds["T14"])

        # Stack along the last dimension
        return np.stack([r, g, b], axis=-1)

    def process_record(self, row):
        """
        Loads raw bands and generates the 6-channel input tensor.

        Structure:
        - Channels 1-3: Ash Composite at t=4 (Current)
        - Channels 4-6: Ash Composite (t=4) - Ash Composite (t=3) (Temporal Diff)
        """
        # Load raw bands (Shape: H x W x T, where T=8)
        # We need bands 11, 14, 15
        try:
            b11 = np.load(os.path.join(CFG.input_dir, row["band_11"]))
            b14 = np.load(os.path.join(CFG.input_dir, row["band_14"]))
            b15 = np.load(os.path.join(CFG.input_dir, row["band_15"]))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Missing band file for record {row['record_id']}: {e}"
            )

        # Indices: t=4 is labeled frame, t=3 is previous frame
        t_curr = 4
        t_prev = 3

        # Compute Ash Composites
        ash_curr = self.get_ash_composite(
            b11[..., t_curr], b14[..., t_curr], b15[..., t_curr]
        )
        ash_prev = self.get_ash_composite(
            b11[..., t_prev], b14[..., t_prev], b15[..., t_prev]
        )

        # Compute Temporal Difference
        diff = ash_curr - ash_prev

        # Concatenate to 6 channels: (H, W, 6)
        img = np.concatenate([ash_curr, diff], axis=-1)

        # Clip to ensure stability (though normalization should handle it)
        img = np.clip(
            img, 0, 1
        )  # Keeping diff in 0-1 range might clip info if diff is negative?
        # Actually, diff can be negative. Standard Ash is [0,1]. Diff is [-1, 1].
        # We should NOT clip the difference blindly to [0,1] or we lose negative changes.
        # However, for the Ash channels (first 3), they are [0,1].
        # Let's re-assemble carefully.

        ash_curr = np.clip(ash_curr, 0, 1)
        # For diff, we keep it as is. Neural nets handle negative inputs fine.
        # But we must ensure the saved numpy file preserves this.

        img = np.concatenate([ash_curr, diff], axis=-1)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = row["record_id"]

        # -----------------------------------------------------------
        # Caching Mechanism
        # -----------------------------------------------------------
        image = None
        cache_path = os.path.join(self.cache_dir, f"{record_id}.npy")

        if self.load_cached_data and os.path.exists(cache_path):
            try:
                image = np.load(cache_path)
            except Exception:
                # If load fails, fall through to re-compute
                pass

        if image is None:
            # Compute from scratch
            image = self.process_record(row)

            # Save to cache if enabled
            if self.load_cached_data:
                np.save(cache_path, image)

        # -----------------------------------------------------------
        # Mask Loading
        # -----------------------------------------------------------
        mask = None
        if self.split in ["train", "validation"]:
            mask_path = os.path.join(CFG.input_dir, row["human_pixel_masks"])
            if os.path.exists(mask_path):
                mask = np.load(mask_path)  # Shape: (H, W, 1)
                mask = mask.astype(np.float32)
            else:
                # Fallback for missing mask (should not happen based on EDA)
                mask = np.zeros((CFG.image_size, CFG.image_size, 1), dtype=np.float32)

        # -----------------------------------------------------------
        # Augmentation
        # -----------------------------------------------------------
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]
        else:
            # Fallback manual conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1))
            if mask is not None:
                mask = torch.from_numpy(mask.transpose(2, 0, 1))

        # -----------------------------------------------------------
        # Return
        # -----------------------------------------------------------
        result = {"image": image, "record_id": record_id}

        if mask is not None:
            # Ensure mask is (C, H, W)
            # Fix: Explicitly permute (H, W, C) -> (C, H, W) if needed
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.permute(2, 0, 1)
            # Ensure mask is (1, H, W)
            elif mask.ndim == 2:
                mask = mask.unsqueeze(0)
            result["mask"] = mask

        return result
