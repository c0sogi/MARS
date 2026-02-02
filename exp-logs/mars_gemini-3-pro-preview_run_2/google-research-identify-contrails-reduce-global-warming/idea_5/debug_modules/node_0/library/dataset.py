import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class ContrailsDataset(Dataset):
    """
    Dataset class for Contrail Identification.
    Generates a 6-channel input tensor:
    - Channels 1-3: Ash False Color Composite at t=4 (labeled frame).
    - Channels 4-6: Temporal Difference of Ash Composite (t=4 minus t=3).
    """

    def __init__(self, metadata, train=True, transform=None):
        """
        Args:
            metadata (pd.DataFrame): Dataframe containing file paths and record_ids.
            train (bool): If True, loads masks and applies augmentations.
            transform (A.Compose, optional): Albumentations transforms. If None, default
                                             affine transforms are used for training.
        """
        self.metadata = metadata
        self.train = train

        # Define default transforms if not provided
        if transform is None:
            if self.train:
                self.transform = A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.ShiftScaleRotate(
                            shift_limit=0.0625,
                            scale_limit=0.1,
                            rotate_limit=45,
                            p=0.5,
                            border_mode=0,  # Constant padding (0)
                        ),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose([ToTensorV2()])
        else:
            self.transform = transform

        # Ash Composite Normalization Bounds
        # (min, max) for each component
        self._ash_bounds = {
            "red": (-6.7, 2.6),  # T15 - T14
            "green": (-6.0, 6.3),  # T14 - T11
            "blue": (243, 303),  # T14
        }

    def __len__(self):
        return len(self.metadata)

    def _normalize(self, data, bounds):
        """
        Linearly normalizes data from [min, max] to [0, 1].
        Clips values outside the range.
        """
        return (np.clip(data, bounds[0], bounds[1]) - bounds[0]) / (
            bounds[1] - bounds[0]
        )

    def _get_ash_composite(self, b11, b14, b15):
        """
        Computes the Ash False Color Composite.
        Args:
            b11, b14, b15: Numpy arrays for bands 11, 14, 15.
        Returns:
            np.ndarray: Normalized Ash composite of shape (H, W, 3).
        """
        # Red: T15 - T14
        r = self._normalize(b15 - b14, self._ash_bounds["red"])

        # Green: T14 - T11
        g = self._normalize(b14 - b11, self._ash_bounds["green"])

        # Blue: T14
        b = self._normalize(b14, self._ash_bounds["blue"])

        # Stack along the last dimension -> (H, W, 3)
        return np.stack([r, g, b], axis=-1)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        record_id = str(row["record_id"])

        # 1. Load Satellite Bands
        # We need bands 11, 14, 15 for the Ash composite
        # Paths are relative in metadata, need to prepend INPUT_DIR
        try:
            path_b11 = os.path.join(Config.INPUT_DIR, row["band_11"])
            path_b14 = os.path.join(Config.INPUT_DIR, row["band_14"])
            path_b15 = os.path.join(Config.INPUT_DIR, row["band_15"])

            # Shape: (H, W, T) where T = 8 (4 before, 1 current, 3 after)
            # Indices: 0,1,2,3 (before), 4 (current), 5,6,7 (after)
            band11 = np.load(path_b11)
            band14 = np.load(path_b14)
            band15 = np.load(path_b15)
        except Exception as e:
            # Fallback for missing files (should not happen based on validation)
            print(f"Error loading bands for {record_id}: {e}")
            # Return zeros if failed
            dummy_img = torch.zeros((6, Config.IMG_SIZE, Config.IMG_SIZE))
            dummy_mask = torch.zeros((1, Config.IMG_SIZE, Config.IMG_SIZE))
            return {"image": dummy_img, "mask": dummy_mask, "record_id": record_id}

        # 2. Extract Temporal Frames
        # T=4 is the labeled frame
        # T=3 is the immediate previous frame
        idx_t4 = 4
        idx_t3 = 3

        b11_t4, b14_t4, b15_t4 = (
            band11[..., idx_t4],
            band14[..., idx_t4],
            band15[..., idx_t4],
        )
        b11_t3, b14_t3, b15_t3 = (
            band11[..., idx_t3],
            band14[..., idx_t3],
            band15[..., idx_t3],
        )

        # 3. Compute Ash Composites
        ash_t4 = self._get_ash_composite(b11_t4, b14_t4, b15_t4)  # (H, W, 3)
        ash_t3 = self._get_ash_composite(b11_t3, b14_t3, b15_t3)  # (H, W, 3)

        # 4. Construct 6-Channel Input
        # Channels 1-3: Ash at t=4
        # Channels 4-6: Ash at t=4 - Ash at t=3
        diff = ash_t4 - ash_t3

        # Concatenate along channel dimension (axis 2)
        # Result shape: (H, W, 6)
        image = np.concatenate([ash_t4, diff], axis=-1)

        # 5. Load Mask (if training/validation)
        mask = None
        if self.train:
            mask_path = os.path.join(Config.INPUT_DIR, row["human_pixel_masks"])
            # Shape: (H, W, 1)
            mask = np.load(mask_path)
        else:
            # For test set, create dummy mask
            mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)

        # 6. Apply Augmentations
        # Albumentations expects 'image' and 'mask'
        # image should be (H, W, C), mask should be (H, W, 1) or (H, W)
        if self.transform:
            augmented = self.transform(
                image=image.astype(np.float32), mask=mask.astype(np.float32)
            )
            image_tensor = augmented["image"]  # (6, H, W)
            mask_tensor = augmented["mask"]  # (1, H, W)

            # Ensure mask is channel-first if not handled by ToTensorV2 correctly for single channel
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            elif mask_tensor.shape[2] == 1 and mask_tensor.ndim == 3:
                # If ToTensorV2 didn't transpose mask (sometimes happens with specific versions)
                mask_tensor = mask_tensor.permute(2, 0, 1)
        else:
            # Manual conversion if no transform provided
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float()

        return {"image": image_tensor, "mask": mask_tensor, "record_id": record_id}
