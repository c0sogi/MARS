import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

from library.config import Config
from library.utils import ash_composite


# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


# ------------------------------------------------------------------------------
# Dataset Implementation
# ------------------------------------------------------------------------------
class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Detection implementing the Temporal Ash-Net strategy.

    Features:
    - Input: 9-channel tensor constructed from 3 time steps (t-1, t, t+1).
    - Channels: Each time step consists of 3 Ash composite channels (Red, Green, Blue).
    - Augmentation: Strictly discrete geometric transforms (Flip, Rotate90) to preserve
      high-frequency contrail details, avoiding interpolation artifacts.
    """

    def __init__(self, metadata_path, stage="train", transform=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file (train/val/test).
            stage (str): One of 'train', 'validation', 'test'.
            transform (albumentations.Compose, optional): Custom augmentation pipeline.
        """
        self.stage = stage
        self.root_dir = Config.INPUT_ROOT

        # Load metadata index
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        self.df = pd.read_csv(metadata_path)

        # Define default transforms if not provided
        if transform is not None:
            self.transform = transform
        else:
            if self.stage == "train":
                # Training: Discrete geometric transforms + ToTensor
                self.transform = A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        ToTensorV2(),
                    ]
                )
            else:
                # Validation/Test: Only ToTensor (HWC -> CHW conversion)
                self.transform = A.Compose(
                    [
                        ToTensorV2(),
                    ]
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Returns:
            dict: {
                'image': torch.Tensor (9, 256, 256),
                'mask': torch.Tensor (1, 256, 256),
                'record_id': str
            }
        """
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # 1. Load Required Spectral Bands
        # We need bands 11, 14, and 15 to compute the Ash composite.
        # The metadata CSV stores relative paths in columns 'band_11', 'band_14', 'band_15'.
        bands_data = {}
        for b in Config.REQUIRED_BANDS:
            col_name = f"band_{b:02d}"
            file_path = os.path.join(self.root_dir, row[col_name])

            try:
                # Load NPY file: Shape is (H, W, T), typically (256, 256, 8)
                bands_data[b] = np.load(file_path)
            except Exception as e:
                # Fallback for data integrity issues
                print(f"Error loading {file_path}: {e}")
                bands_data[b] = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, 8), dtype=np.float32
                )

        # 2. Construct Temporal Ash Composite
        # We extract time steps t-1 (index 3), t (index 4), and t+1 (index 5).
        temporal_slices = []

        for t_idx in Config.TEMPORAL_INDICES:
            # Extract the specific time slice from the temporal dimension
            # Resulting shape for each band: (H, W)
            b11 = bands_data[11][..., t_idx]
            b14 = bands_data[14][..., t_idx]
            b15 = bands_data[15][..., t_idx]

            # Compute Ash Composite for this time step
            # Returns (H, W, 3) normalized to [0, 1]
            ash = ash_composite(b11, b14, b15)
            temporal_slices.append(ash)

        # Concatenate along the channel dimension to form the input tensor
        # [Ash_t-1, Ash_t, Ash_t+1] -> Shape (H, W, 9)
        image = np.concatenate(temporal_slices, axis=-1).astype(np.float32)

        # 3. Load Ground Truth Mask
        mask = None
        if self.stage in ["train", "validation"]:
            mask_col = "human_pixel_masks"
            # Check if mask path exists in the dataframe row
            if mask_col in row and pd.notna(row[mask_col]):
                mask_path = os.path.join(self.root_dir, row[mask_col])
                try:
                    # Load mask: Shape (H, W, 1)
                    mask = np.load(mask_path).astype(np.float32)
                except Exception as e:
                    print(f"Error loading mask {mask_path}: {e}")
                    mask = np.zeros(
                        (Config.IMG_SIZE, Config.IMG_SIZE, 1), dtype=np.float32
                    )
            else:
                # Fallback if mask is missing (e.g. some validation sets)
                mask = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 1), dtype=np.float32)
        else:
            # Test stage: Create a dummy mask for compatibility with transforms
            mask = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 1), dtype=np.float32)

        # 4. Apply Augmentations
        if self.transform:
            # Albumentations expects 'image' (H, W, C) and 'mask' (H, W, C)
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 5. Return Dictionary
        # 'image': Tensor (C, H, W) -> (9, 256, 256)
        # 'mask': Tensor (C, H, W) -> (1, 256, 256)
        return {"image": image, "mask": mask, "record_id": record_id}
