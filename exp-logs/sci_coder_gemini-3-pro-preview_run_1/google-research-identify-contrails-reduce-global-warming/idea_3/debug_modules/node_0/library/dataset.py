import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import load_metadata


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Detection.
    Implements on-the-fly 'Ash' composite generation and augmentation.
    """

    def __init__(self, metadata, split="train", transform=None):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing record_ids and file paths.
            split (str): One of 'train', 'validation', 'test'.
            transform (albumentations.Compose): Transforms to apply.
        """
        self.metadata = metadata
        self.split = split
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

        # Temporal index for the labeled frame (n_times_before = 4)
        self.t_index = 4

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        record_id = str(row["record_id"])

        # ----------------------------------------------------------------------
        # 1. Load Spectral Bands
        # ----------------------------------------------------------------------
        # We need Band 11, 14, 15 for the Ash composite.
        # Files are H x W x T. We extract the specific time step.

        try:
            p11 = os.path.join(self.input_dir, row["band_11"])
            p14 = os.path.join(self.input_dir, row["band_14"])
            p15 = os.path.join(self.input_dir, row["band_15"])

            # Load only the required timestep to save I/O and memory
            # Using mmap_mode='r' could be an option, but direct load + slice is usually fine for this size
            b11 = np.load(p11)[:, :, self.t_index].astype(np.float32)
            b14 = np.load(p14)[:, :, self.t_index].astype(np.float32)
            b15 = np.load(p15)[:, :, self.t_index].astype(np.float32)

        except Exception as e:
            print(f"Error loading record {record_id}: {e}")
            # Fallback to zeros to prevent crash
            sz = Config.IMAGE_SIZE
            b11 = np.zeros((sz, sz), dtype=np.float32)
            b14 = np.zeros((sz, sz), dtype=np.float32)
            b15 = np.zeros((sz, sz), dtype=np.float32)

        # ----------------------------------------------------------------------
        # 2. Compute Ash Composite
        # ----------------------------------------------------------------------
        # Red: Band 15 - Band 14
        r = b15 - b14

        # Green: Band 14 - Band 11
        g = b14 - b11

        # Blue: Band 14
        b = b14

        # ----------------------------------------------------------------------
        # 3. Normalize
        # ----------------------------------------------------------------------
        # Apply min-max normalization based on Config constants
        r = (r - Config.ASH_RED_MIN) / (Config.ASH_RED_MAX - Config.ASH_RED_MIN)
        g = (g - Config.ASH_GREEN_MIN) / (Config.ASH_GREEN_MAX - Config.ASH_GREEN_MIN)
        b = (b - Config.ASH_BLUE_MIN) / (Config.ASH_BLUE_MAX - Config.ASH_BLUE_MIN)

        # Clip to [0, 1] range
        r = np.clip(r, 0, 1)
        g = np.clip(g, 0, 1)
        b = np.clip(b, 0, 1)

        # Stack to (H, W, 3)
        image = np.dstack([r, g, b])

        # ----------------------------------------------------------------------
        # 4. Load Mask (if applicable)
        # ----------------------------------------------------------------------
        mask = None
        if self.split in ["train", "validation"]:
            mask_path_rel = row.get("human_pixel_masks")
            if pd.notna(mask_path_rel):
                mask_path = os.path.join(self.input_dir, mask_path_rel)
                # Mask shape is (H, W, 1)
                mask = np.load(mask_path).astype(np.float32)
            else:
                # Fallback if path is missing in metadata
                mask = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1), dtype=np.float32
                )

        # ----------------------------------------------------------------------
        # 5. Apply Transforms
        # ----------------------------------------------------------------------
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]
        else:
            # Manual conversion if no transform provided (fallback)
            image = torch.from_numpy(image.transpose(2, 0, 1))
            if mask is not None:
                mask = torch.from_numpy(mask.transpose(2, 0, 1))

        # ----------------------------------------------------------------------
        # 6. Return
        # ----------------------------------------------------------------------
        if self.split in ["train", "validation"]:
            return image, mask
        else:
            # For test, we need the record_id for submission
            return image, record_id


def get_transforms(split="train"):
    """
    Returns the albumentations transforms for the specified split.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        debug (bool): If True, subsamples the dataset.
        debug_sample_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    train_meta = load_metadata("train")
    val_meta = load_metadata("validation")
    test_meta = load_metadata("test")

    # 2. Debug Subsampling
    if debug:
        print(f"Debug mode enabled. Subsampling to {debug_sample_size} records.")
        train_meta = train_meta.iloc[:debug_sample_size]
        val_meta = val_meta.iloc[:debug_sample_size]
        test_meta = test_meta.iloc[:debug_sample_size]

    # 3. Define Transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("validation")
    test_transform = get_transforms("test")

    # 4. Create Datasets
    train_dataset = ContrailDataset(
        train_meta, split="train", transform=train_transform
    )
    val_dataset = ContrailDataset(val_meta, split="validation", transform=val_transform)
    test_dataset = ContrailDataset(test_meta, split="test", transform=test_transform)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
