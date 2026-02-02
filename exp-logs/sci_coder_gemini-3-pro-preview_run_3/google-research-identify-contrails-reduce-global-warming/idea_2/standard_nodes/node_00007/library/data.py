import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# --- Constants for Ash Composite Normalization ---
# Based on standard GOES-16 Ash RGB recipes and competition heuristics.
# Red:   Band 15 - Band 14
# Green: Band 14 - Band 11
# Blue:  Band 14
ASH_BOUNDS = {"R": (-6.7, 2.6), "G": (-6.3, 12.8), "B": (243, 303)}


def normalize_range(data, min_v, max_v):
    """
    Normalizes data to [0, 1] range based on min/max bounds and clips values.
    """
    norm = (data - min_v) / (max_v - min_v)
    return np.clip(norm, 0, 1)


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Segmentation.
    Loads satellite bands, constructs Ash composite + Temporal Difference features.
    """

    def __init__(self, df, mode="train", transforms=None, root_dir=Config.INPUT_ROOT):
        self.df = df
        self.mode = mode
        self.transforms = transforms
        self.root_dir = root_dir

        # Pre-check columns to avoid errors in __getitem__
        self.has_mask = "human_pixel_masks" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # Load required bands: 11, 14, 15
        # Paths are relative in CSV, so join with root
        bands = {}
        for b in [11, 14, 15]:
            # Handle potential float/int formatting in column names if necessary,
            # but usually they match the CSV header 'band_xx'
            col_name = f"band_{b:02d}"
            path = os.path.join(self.root_dir, row[col_name])
            try:
                bands[b] = np.load(path)  # Shape: H x W x T
            except Exception as e:
                # Fallback for robustness (should not happen given metadata check)
                print(f"Error loading {path}: {e}")
                # Return dummy zeros
                bands[b] = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 8), dtype=np.float32
                )

        # Time indices
        # n_times_before=4 means index 4 is the current labeled frame
        t_curr = Config.N_TIMES_BEFORE
        t_prev = t_curr - 1

        def get_ash_composite(t_idx):
            # Extract bands at specific time step
            t11 = bands[11][..., t_idx]
            t14 = bands[14][..., t_idx]
            t15 = bands[15][..., t_idx]

            # Compute Ash channels
            # R = T15 - T14
            r = normalize_range(t15 - t14, ASH_BOUNDS["R"][0], ASH_BOUNDS["R"][1])
            # G = T14 - T11
            g = normalize_range(t14 - t11, ASH_BOUNDS["G"][0], ASH_BOUNDS["G"][1])
            # B = T14
            b = normalize_range(t14, ASH_BOUNDS["B"][0], ASH_BOUNDS["B"][1])

            return np.stack([r, g, b], axis=-1)  # H x W x 3

        # Compute features
        ash_curr = get_ash_composite(t_curr)
        ash_prev = get_ash_composite(t_prev)
        ash_diff = ash_curr - ash_prev  # Range approx [-1, 1]

        # Concatenate to 6 channels: [Ash_t, Ash_t - Ash_{t-1}]
        image = np.concatenate([ash_curr, ash_diff], axis=-1)  # H x W x 6

        # Ensure float32
        image = image.astype(np.float32)

        if self.mode in ["train", "validation"]:
            # Load Mask
            if self.has_mask:
                mask_path = os.path.join(self.root_dir, row["human_pixel_masks"])
                mask = np.load(mask_path)  # H x W x 1
                mask = mask.astype(np.float32)
            else:
                # Fallback if mask missing in val (unlikely)
                mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)

            # Apply Augmentations
            if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                # If no transforms, manually convert to tensor
                image = torch.from_numpy(image.transpose(2, 0, 1))
                mask = torch.from_numpy(mask.transpose(2, 0, 1))

            return image, mask

        else:
            # Test mode
            if self.transforms:
                augmented = self.transforms(image=image)
                image = augmented["image"]
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1))

            return image, record_id


def get_transforms(split):
    """
    Returns Albumentations transforms for the specified split.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
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
        # Validation and Test
        return A.Compose([ToTensorV2(transpose_mask=True)])


def get_loaders(debug=False, batch_size=Config.BATCH_SIZE):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, subsets the data for quick debugging.
        batch_size (int): Batch size for loaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # Debug Subsampling
    if debug:
        df_train = df_train.sample(
            n=min(len(df_train), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        # We usually keep test set intact or small sample for debug pipeline check
        df_test = df_test.sample(
            n=min(len(df_test), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Datasets
    train_ds = ContrailDataset(
        df_train, mode="train", transforms=get_transforms("train")
    )

    val_ds = ContrailDataset(
        df_val, mode="validation", transforms=get_transforms("validation")
    )

    test_ds = ContrailDataset(df_test, mode="test", transforms=get_transforms("test"))

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
