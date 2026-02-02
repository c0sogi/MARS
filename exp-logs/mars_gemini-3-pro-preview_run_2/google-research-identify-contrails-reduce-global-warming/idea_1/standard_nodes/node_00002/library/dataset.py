import os
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Segmentation.

    Loads satellite bands, constructs an Ash False Color Composite,
    and computes a temporal difference feature.

    Input: 6 Channels
        - Channels 0-2: Ash Composite (t=4)
        - Channels 3-5: Difference (Ash t=4 - Ash t=3)

    Target: Binary Mask (1 Channel)
    """

    def __init__(self, metadata_df, split="train", transform=None):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing file paths.
            split (str): 'train', 'validation', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = metadata_df
        self.split = split
        self.transform = transform

        # Pre-compute full paths to avoid os.path.join overhead in __getitem__
        # We need bands 11, 14, 15
        self.band_11_paths = [
            os.path.join(Config.INPUT_DIR, p) for p in self.df["band_11"].values
        ]
        self.band_14_paths = [
            os.path.join(Config.INPUT_DIR, p) for p in self.df["band_14"].values
        ]
        self.band_15_paths = [
            os.path.join(Config.INPUT_DIR, p) for p in self.df["band_15"].values
        ]

        # Load mask paths if available
        self.mask_paths = None
        if "human_pixel_masks" in self.df.columns:
            self.mask_paths = [
                os.path.join(Config.INPUT_DIR, p)
                for p in self.df["human_pixel_masks"].values
            ]

    def __len__(self):
        return len(self.df)

    @staticmethod
    def normalize_range(data, min_val, max_val):
        """
        Linearly normalizes data to [0, 1] based on provided min/max.
        Clips values outside the range.
        """
        return np.clip((data - min_val) / (max_val - min_val), 0, 1)

    def get_ash_vector(self, idx, time_idx):
        """
        Generates the Ash False Color Composite for a specific time step.

        Args:
            idx (int): Index of the sample in the dataframe.
            time_idx (int): Temporal index to extract (e.g., 4 or 3).

        Returns:
            np.ndarray: Normalized Ash composite of shape (H, W, 3).
        """
        # Load specific bands.
        # NPY files are shape (H, W, T). We slice [:, :, time_idx] immediately.
        # Using mmap_mode='r' can be faster if files are large, but these are small chunks.
        # Direct load is usually fine for this size (2MB).

        try:
            band11 = np.load(self.band_11_paths[idx])[:, :, time_idx]
            band14 = np.load(self.band_14_paths[idx])[:, :, time_idx]
            band15 = np.load(self.band_15_paths[idx])[:, :, time_idx]
        except Exception as e:
            # Fallback for potential file read errors, though paths are validated
            print(f"Error loading bands for index {idx}: {e}")
            # Return zeros as fallback
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

        # Compute components
        # Red: Band 15 - Band 14
        r = self.normalize_range(
            band15 - band14, Config.ASH_RED_MIN, Config.ASH_RED_MAX
        )

        # Green: Band 14 - Band 11
        g = self.normalize_range(
            band14 - band11, Config.ASH_GREEN_MIN, Config.ASH_GREEN_MAX
        )

        # Blue: Band 14
        b = self.normalize_range(band14, Config.ASH_BLUE_MIN, Config.ASH_BLUE_MAX)

        # Stack to (H, W, 3)
        return np.stack([r, g, b], axis=-1).astype(np.float32)

    def __getitem__(self, idx):
        # 1. Generate Ash Composite for Current Frame (t=4)
        ash_curr = self.get_ash_vector(idx, Config.LABELED_FRAME_IDX)

        # 2. Generate Ash Composite for Previous Frame (t=3)
        ash_prev = self.get_ash_vector(idx, Config.PREV_FRAME_IDX)

        # 3. Compute Temporal Difference
        # Simple subtraction. Since inputs are [0,1], diff is [-1, 1].
        # We don't necessarily need to shift/scale it for NN, but keeping it raw is fine.
        # Some implementations add 0.5 to center at 0.5, but raw diff is informative.
        diff = ash_curr - ash_prev

        # 4. Concatenate to form 6-channel input
        # Shape: (H, W, 6)
        image = np.concatenate([ash_curr, diff], axis=-1)

        # 5. Load Mask (if available)
        mask = None
        if self.mask_paths is not None:
            try:
                # Shape: (H, W, 1)
                mask = np.load(self.mask_paths[idx]).astype(np.float32)
            except Exception as e:
                print(f"Error loading mask for index {idx}: {e}")
                mask = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 1), dtype=np.float32)

        # 6. Apply Transforms (if any)
        # Albumentations expects 'image' and 'mask'
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # 7. Convert to Tensor
        # Transpose from (H, W, C) to (C, H, W) for PyTorch
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        if mask is not None:
            # Transpose mask from (H, W, 1) to (1, H, W)
            # If transform converted it to tensor already, handle that,
            # but usually albumentations returns numpy.
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).permute(2, 0, 1).float()
            elif (
                isinstance(mask, torch.Tensor) and mask.ndim == 3 and mask.shape[2] == 1
            ):
                mask = mask.permute(2, 0, 1).float()

            return image, mask

        return image
