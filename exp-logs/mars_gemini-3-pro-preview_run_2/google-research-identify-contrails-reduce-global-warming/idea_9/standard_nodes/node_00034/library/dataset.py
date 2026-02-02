import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# ==========================================
# Normalization Constants
# ==========================================
# Ash Composite Bounds (Kelvin)
# Derived from GOES-16 ABI Band specifications and domain heuristics for Ash/Contrails
BOUNDS_ASH_R = (-6.7, 2.6)  # Band 15 - Band 14
BOUNDS_ASH_G = (-6.3, 5.7)  # Band 14 - Band 11
BOUNDS_ASH_B = (243.6, 303.2)  # Band 14

# Temporal Difference Bounds (Kelvin)
# Differences between t=4 and t=3 are usually small.
# We use a symmetric range to capture subtle changes.
BOUNDS_DIFF = (-2.0, 2.0)


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Detection.
    Loads Bands 11, 14, 15 to construct a 6-channel input:
    - 3 Channels: Ash False Color Composite (t=4)
    - 3 Channels: Temporal Difference (t=4 - t=3)
    """

    def __init__(self, metadata_path, split="train", transform=None, max_samples=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            split (str): 'train', 'validation', or 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
            max_samples (int, optional): Limit dataset size for debugging.
        """
        self.split = split
        self.transform = transform

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Convert record_id to string to ensure consistency
        self.df["record_id"] = self.df["record_id"].astype(str)

        # Debugging: Limit samples if requested
        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def normalize(self, data, bounds):
        """
        Min-Max normalization to [0, 1] based on provided bounds.
        """
        min_v, max_v = bounds
        return (data - min_v) / (max_v - min_v)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = row["record_id"]

        # -----------------------------------------------------------
        # 1. Load Satellite Bands (11, 14, 15)
        # -----------------------------------------------------------
        # Paths are relative in metadata, join with INPUT_DIR
        try:
            # Shape of loaded npy: (H, W, T) where T=8
            # t=0..3 (before), t=4 (labeled), t=5..7 (after)
            # We need t=3 (prev) and t=4 (current)

            path_11 = os.path.join(Config.INPUT_DIR, row["band_11"])
            path_14 = os.path.join(Config.INPUT_DIR, row["band_14"])
            path_15 = os.path.join(Config.INPUT_DIR, row["band_15"])

            b11 = np.load(path_11)
            b14 = np.load(path_14)
            b15 = np.load(path_15)

            # Extract time steps
            # Current frame (t=4)
            t4_b11 = b11[..., 4]
            t4_b14 = b14[..., 4]
            t4_b15 = b15[..., 4]

            # Previous frame (t=3)
            t3_b11 = b11[..., 3]
            t3_b14 = b14[..., 3]
            t3_b15 = b15[..., 3]

        except Exception as e:
            # Fallback for missing files (should not happen with validated metadata)
            print(f"Error loading bands for {record_id}: {e}")
            # Return zeros of correct shape
            img = torch.zeros(
                (Config.INPUT_CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                dtype=torch.float32,
            )
            if self.split == "test":
                return img, record_id
            else:
                mask = torch.zeros(
                    (1, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=torch.float32
                )
                return img, mask

        # -----------------------------------------------------------
        # 2. Construct Ash Composite (Channels 1-3)
        # -----------------------------------------------------------
        # Red: Band 15 - Band 14
        ash_r = self.normalize(t4_b15 - t4_b14, BOUNDS_ASH_R)

        # Green: Band 14 - Band 11
        ash_g = self.normalize(t4_b14 - t4_b11, BOUNDS_ASH_G)

        # Blue: Band 14
        ash_b = self.normalize(t4_b14, BOUNDS_ASH_B)

        # -----------------------------------------------------------
        # 3. Construct Temporal Difference (Channels 4-6)
        # -----------------------------------------------------------
        # Diff 1: Band 11 (t4 - t3)
        diff_11 = self.normalize(t4_b11 - t3_b11, BOUNDS_DIFF)

        # Diff 2: Band 14 (t4 - t3)
        diff_14 = self.normalize(t4_b14 - t3_b14, BOUNDS_DIFF)

        # Diff 3: Band 15 (t4 - t3)
        diff_15 = self.normalize(t4_b15 - t3_b15, BOUNDS_DIFF)

        # Stack into (H, W, 6)
        image = np.stack([ash_r, ash_g, ash_b, diff_11, diff_14, diff_15], axis=-1)

        # Clip to ensure [0, 1] range after normalization
        image = np.clip(image, 0, 1).astype(np.float32)

        # -----------------------------------------------------------
        # 4. Load Mask (if available)
        # -----------------------------------------------------------
        mask = None
        if self.split in ["train", "validation"]:
            try:
                mask_path = os.path.join(Config.INPUT_DIR, row["human_pixel_masks"])
                mask = np.load(mask_path)  # Shape (H, W, 1)
                mask = mask.astype(np.float32)
            except Exception as e:
                print(f"Error loading mask for {record_id}: {e}")
                mask = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1), dtype=np.float32
                )

        # -----------------------------------------------------------
        # 5. Augmentations
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
            # Manual ToTensor if no transform provided
            image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
            if mask is not None:
                mask = torch.from_numpy(mask).permute(2, 0, 1)  # (C, H, W)

        # -----------------------------------------------------------
        # 6. Return
        # -----------------------------------------------------------
        if self.split == "test":
            # For test, we might need record_id for submission
            return image, record_id
        else:
            return image, mask


def get_transforms(split="train"):
    """
    Returns albumentations transform pipeline.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Affine transformations only - no elastic/grid distortions
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=0,
                ),
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        # Validation / Test
        return A.Compose([ToTensorV2(transpose_mask=True)])


def get_dataloader(
    split,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    max_samples=None,
):
    """
    Factory function to create DataLoaders.

    Args:
        split (str): 'train', 'validation', or 'test'.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        max_samples (int, optional): Limit dataset size.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # Determine metadata path based on split
    if split == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
        shuffle = True
    elif split == "validation":
        metadata_path = Config.VAL_METADATA_PATH
        shuffle = False
    elif split == "test":
        metadata_path = Config.TEST_METADATA_PATH
        shuffle = False
    else:
        raise ValueError(f"Unknown split: {split}")

    # Get transforms
    transform = get_transforms(split)

    # Create Dataset
    dataset = ContrailDataset(
        metadata_path=metadata_path,
        split=split,
        transform=transform,
        max_samples=max_samples,
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),  # Drop last incomplete batch only for training
    )

    return dataloader
