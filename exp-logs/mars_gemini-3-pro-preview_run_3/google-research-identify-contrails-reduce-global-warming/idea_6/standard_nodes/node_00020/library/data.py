import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# --- Constants ---
# Indices in the temporal sequence (0-based)
# n_times_before = 4, so the labeled frame is index 4.
T_CURRENT = 4
T_PREV = 3

# Bounds for Ash Color Scheme Normalization (Kelvin)
# Based on standard GOES-16 Ash RGB recipes
ASH_BOUNDS = {
    "15_14": (-4.0, 2.0),  # Red component: T15 - T14
    "14_11": (-4.0, 5.0),  # Green component: T14 - T11
    "14": (243.0, 303.0),  # Blue component: T14
}


def normalize_range(data, bounds):
    """
    Normalizes data to the [0, 1] range based on min/max bounds.
    """
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def get_ash_composite(band_11, band_14, band_15):
    """
    Constructs the 3-channel Ash False Color composite from raw brightness temperatures.

    Args:
        band_11 (np.ndarray): Band 11 data (H, W)
        band_14 (np.ndarray): Band 14 data (H, W)
        band_15 (np.ndarray): Band 15 data (H, W)

    Returns:
        np.ndarray: Normalized Ash composite of shape (H, W, 3) in range [0, 1].
    """
    # Calculate components
    r = band_15 - band_14
    g = band_14 - band_11
    b = band_14

    # Normalize
    r_norm = normalize_range(r, ASH_BOUNDS["15_14"])
    g_norm = normalize_range(g, ASH_BOUNDS["14_11"])
    b_norm = normalize_range(b, ASH_BOUNDS["14"])

    # Stack and clip
    ash_img = np.stack([r_norm, g_norm, b_norm], axis=-1)
    ash_img = np.clip(ash_img, 0, 1)

    return ash_img


def get_transforms(split):
    """
    Returns the Albumentations transformation pipeline for a given split.

    Args:
        split (str): 'train', 'validation', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if split == "train":
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Normalize with ImageNet stats as we use pretrained backbones
                A.Normalize(
                    mean=[0.485, 0.456, 0.406] * 2,  # Applied to 6 channels
                    std=[0.229, 0.224, 0.225] * 2,
                    max_pixel_value=1.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406] * 2,
                    std=[0.229, 0.224, 0.225] * 2,
                    max_pixel_value=1.0,
                ),
                ToTensorV2(),
            ]
        )


class ContrailsDataset(Dataset):
    """
    PyTorch Dataset for Contrail Segmentation.
    Loads satellite bands, constructs the 6-channel input tensor, and returns masks.
    """

    def __init__(self, metadata_df, split="train", transform=None):
        self.df = metadata_df
        self.split = split
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

        # Pre-filter necessary columns to avoid overhead
        self.records = self.df.to_dict("records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        record_id = str(row["record_id"])

        # --- Load Bands ---
        # We only need Bands 11, 14, 15 for the Ash composite.
        # Paths are relative in the CSV, e.g., "train/ID/band_11.npy"
        try:
            p11 = os.path.join(self.input_dir, row["band_11"])
            p14 = os.path.join(self.input_dir, row["band_14"])
            p15 = os.path.join(self.input_dir, row["band_15"])

            # Load full temporal sequences: (H, W, T)
            b11_seq = np.load(p11)
            b14_seq = np.load(p14)
            b15_seq = np.load(p15)

            # Extract time steps t (current) and t-1 (previous)
            # Shape becomes (H, W)
            b11_t = b11_seq[..., T_CURRENT]
            b14_t = b14_seq[..., T_CURRENT]
            b15_t = b15_seq[..., T_CURRENT]

            b11_tm1 = b11_seq[..., T_PREV]
            b14_tm1 = b14_seq[..., T_PREV]
            b15_tm1 = b15_seq[..., T_PREV]

        except Exception as e:
            # Fallback for corrupt data (should not happen in clean dataset)
            print(f"Error loading bands for {record_id}: {e}")
            # Return dummy zero tensors
            img = torch.zeros((Config.N_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE))
            if self.split != "test":
                mask = torch.zeros((1, Config.IMG_SIZE, Config.IMG_SIZE))
                return img, mask
            return img

        # --- Construct Input Tensor ---
        # 1. Ash Composite at time t
        ash_t = get_ash_composite(b11_t, b14_t, b15_t)  # (H, W, 3)

        # 2. Ash Composite at time t-1
        ash_tm1 = get_ash_composite(b11_tm1, b14_tm1, b15_tm1)  # (H, W, 3)

        # 3. Temporal Difference
        diff = ash_t - ash_tm1

        # 4. Concatenate: Ash_t (3ch) + Diff (3ch) = 6 Channels
        # Shape: (H, W, 6)
        img = np.concatenate([ash_t, diff], axis=-1)

        # Ensure float32
        img = img.astype(np.float32)

        # --- Load Mask (if available) ---
        mask = None
        if self.split in ["train", "validation"]:
            mask_path = os.path.join(self.input_dir, row["human_pixel_masks"])
            # Mask shape: (H, W, 1)
            mask = np.load(mask_path).astype(np.float32)

        # --- Apply Transforms ---
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
                # Ensure mask is channel-first (1, H, W)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.shape[2] == 1:  # If (H, W, 1) came back
                    mask = mask.permute(2, 0, 1)
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]

        if self.split == "test":
            return img, record_id
        else:
            return img, mask


def get_loaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=False
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        debug (bool): If True, subsamples the dataset for quick debugging.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders.
    """
    # Load Metadata
    train_csv = os.path.join(Config.METADATA_DIR, "train.csv")
    val_csv = os.path.join(Config.METADATA_DIR, "validation.csv")
    test_csv = os.path.join(Config.METADATA_DIR, "test.csv")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # Debug Mode: Subsample
    if debug:
        df_train = df_train.head(batch_size * 2)
        df_val = df_val.head(batch_size * 2)
        df_test = df_test.head(batch_size * 2)

    # Transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("validation")
    test_transform = get_transforms("test")

    # Datasets
    train_dataset = ContrailsDataset(df_train, split="train", transform=train_transform)
    val_dataset = ContrailsDataset(df_val, split="validation", transform=val_transform)
    test_dataset = ContrailsDataset(df_test, split="test", transform=test_transform)

    # DataLoaders
    # Note: drop_last=True for train to maintain consistent batch stats
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
