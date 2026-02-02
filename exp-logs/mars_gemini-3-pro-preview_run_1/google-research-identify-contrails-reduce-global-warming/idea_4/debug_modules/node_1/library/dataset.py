import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------


def normalize_range(data, min_val, max_val):
    """
    Normalizes data to [0, 1] based on provided min/max bounds.
    Clips values outside the range.
    """
    data = np.clip(data, min_val, max_val)
    data = (data - min_val) / (max_val - min_val)
    return data


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline based on the mode.

    Args:
        mode (str): 'train', 'validation', or 'test'.

    Returns:
        A.Compose: Transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


# ------------------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------------------


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Detection.

    Loads specific infrared bands (11, 14, 15), computes the Ash False-Color Composite,
    and applies normalization and augmentation.
    """

    def __init__(self, metadata_csv, mode="train", transform=None, debug=False):
        """
        Args:
            metadata_csv (str): Path to the metadata CSV file (train/val/test).
            mode (str): 'train', 'validation', or 'test'.
            transform (A.Compose): Albumentations transforms.
            debug (bool): If True, limits the dataset size for debugging.
        """
        self.mode = mode
        self.transform = transform
        self.root_dir = Config.INPUT_DIR

        # Load metadata
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

        self.df = pd.read_csv(metadata_csv)

        # Handle string conversion for record_id to ensure consistency
        self.df["record_id"] = self.df["record_id"].astype(str)

        # Debugging: Sample subset
        if debug:
            sample_size = min(len(self.df), Config.DEBUG_SAMPLE_SIZE)
            self.df = self.df.sample(
                n=sample_size, random_state=Config.SEED
            ).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = row["record_id"]

        # ---------------------------------------------------------------------
        # 1. Load Infrared Bands
        # ---------------------------------------------------------------------
        # We need bands 11, 14, and 15.
        # The files are NPY arrays of shape (H, W, T).
        # We need the (n_times_before + 1)-th image, which is index 4.

        try:
            # Construct full paths
            p11 = os.path.join(self.root_dir, row["band_11"])
            p14 = os.path.join(self.root_dir, row["band_14"])
            p15 = os.path.join(self.root_dir, row["band_15"])

            # Load and slice temporal dimension (index 4)
            # Using mmap_mode='r' can be faster for slicing large files without full read
            b11 = np.load(p11, mmap_mode="r")[:, :, 4].astype(np.float32)
            b14 = np.load(p14, mmap_mode="r")[:, :, 4].astype(np.float32)
            b15 = np.load(p15, mmap_mode="r")[:, :, 4].astype(np.float32)

        except Exception as e:
            # Fallback for missing files or corruption (should not happen in clean data)
            print(f"Error loading bands for {record_id}: {e}")
            H, W = Config.IMAGE_SIZE, Config.IMAGE_SIZE
            b11 = np.zeros((H, W), dtype=np.float32)
            b14 = np.zeros((H, W), dtype=np.float32)
            b15 = np.zeros((H, W), dtype=np.float32)

        # ---------------------------------------------------------------------
        # 2. Compute Ash False-Color Composite
        # ---------------------------------------------------------------------
        # Red: Optical Depth (Band 15 - Band 14)
        r_channel = b15 - b14

        # Green: Particle Phase (Band 14 - Band 11)
        g_channel = b14 - b11

        # Blue: Temperature (Band 14)
        b_channel = b14

        # ---------------------------------------------------------------------
        # 3. Normalize
        # ---------------------------------------------------------------------
        r_norm = normalize_range(r_channel, Config.ASH_RED_MIN, Config.ASH_RED_MAX)
        g_norm = normalize_range(g_channel, Config.ASH_GREEN_MIN, Config.ASH_GREEN_MAX)
        b_norm = normalize_range(b_channel, Config.ASH_BLUE_MIN, Config.ASH_BLUE_MAX)

        # Stack to (H, W, 3)
        image = np.stack([r_norm, g_norm, b_norm], axis=-1)

        # ---------------------------------------------------------------------
        # 4. Load Mask (Train/Validation only)
        # ---------------------------------------------------------------------
        mask = None
        if self.mode in ["train", "validation"]:
            mask_path_rel = row.get("human_pixel_masks", None)
            if mask_path_rel and isinstance(mask_path_rel, str):
                full_mask_path = os.path.join(self.root_dir, mask_path_rel)
                # Mask is (H, W, 1) or (H, W)
                mask = np.load(full_mask_path).astype(np.float32)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]  # Convert to (H, W)
            else:
                # Fallback if mask path is missing in metadata
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # ---------------------------------------------------------------------
        # 5. Augmentations
        # ---------------------------------------------------------------------
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
                # Ensure mask has channel dimension for PyTorch (1, H, W)
                mask = mask.unsqueeze(0)
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # ---------------------------------------------------------------------
        # 6. Return
        # ---------------------------------------------------------------------
        # Return a dictionary to handle different modes flexibly
        sample = {"image": image, "record_id": record_id}

        if mask is not None:
            sample["mask"] = mask

        return sample


# ------------------------------------------------------------------------------
# DataLoader Builder
# ------------------------------------------------------------------------------


def get_dataloader(
    mode, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=False
):
    """
    Creates a DataLoader for the specified mode.

    Args:
        mode (str): 'train', 'validation', or 'test'.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        debug (bool): Debug flag.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    # Select CSV path based on mode
    if mode == "train":
        csv_path = Config.TRAIN_CSV
        shuffle = True
        drop_last = True
    elif mode == "validation":
        csv_path = Config.VALIDATION_CSV
        shuffle = False
        drop_last = False
    elif mode == "test":
        csv_path = Config.TEST_CSV
        shuffle = False
        drop_last = False
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Get transforms
    transforms = get_transforms(mode)

    # Create Dataset
    dataset = ContrailDataset(
        metadata_csv=csv_path, mode=mode, transform=transforms, debug=debug
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    return dataloader
