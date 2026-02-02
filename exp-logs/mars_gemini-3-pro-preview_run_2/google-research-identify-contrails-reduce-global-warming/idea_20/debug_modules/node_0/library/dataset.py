import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library import config, utils

# =============================================================================
# CONSTANTS & NORMALIZATION BOUNDS
# =============================================================================
# Ash Color Scheme Bounds
# Red: T15 - T14
ASH_RED_MIN = -4.0
ASH_RED_MAX = 2.0

# Green: T14 - T11
ASH_GREEN_MIN = -4.0
ASH_GREEN_MAX = 5.0

# Blue: T14
ASH_BLUE_MIN = 243.0
ASH_BLUE_MAX = 303.0

# Temporal Indices (0-based)
# n_times_before = 4, so the labeled frame is at index 4
IDX_CURRENT = 4
IDX_PREV = 3


def normalize_range(data, min_val, max_val):
    """
    Normalizes data to [0, 1] based on provided min/max bounds.
    Clips values outside the range.
    """
    return np.clip((data - min_val) / (max_val - min_val), 0, 1)


def get_transforms(stage="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        stage (str): 'train', 'validation', or 'test'.
    """
    if stage == "train":
        return A.Compose(
            [
                # Strict Affine Transformations
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=180,  # Full rotation allowed
                    p=0.5,
                    border_mode=0,  # Constant padding (0)
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # No elastic/grid distortions to preserve linearity
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Just convert to tensor
        return A.Compose([ToTensorV2()])


class ContrailsDataset(Dataset):
    def __init__(self, metadata_path, stage="train", transform=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            stage (str): 'train', 'validation', or 'test'.
            transform (albumentations.Compose): Transformations to apply.
        """
        self.stage = stage
        self.transform = transform

        # Load metadata
        try:
            self.df = pd.read_csv(metadata_path)
        except FileNotFoundError:
            print(
                f"Warning: Metadata file not found at {metadata_path}. Dataset will be empty."
            )
            self.df = pd.DataFrame()

        # Ensure record_id is treated as string
        if not self.df.empty:
            self.df["record_id"] = self.df["record_id"].astype(str)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = row["record_id"]

        # ---------------------------------------------------------------------
        # 1. Load Satellite Bands
        # ---------------------------------------------------------------------
        # We need Bands 11, 14, 15 for Ash Color and Differences
        # Paths in metadata are relative to INPUT_DIR
        try:
            path_b11 = os.path.join(config.INPUT_DIR, row["band_11"])
            path_b14 = os.path.join(config.INPUT_DIR, row["band_14"])
            path_b15 = os.path.join(config.INPUT_DIR, row["band_15"])

            # Load NPY files (H, W, T)
            # T = 8 usually
            b11 = np.load(path_b11)
            b14 = np.load(path_b14)
            b15 = np.load(path_b15)
        except Exception as e:
            # Fallback for missing files (should not happen with validated metadata)
            print(f"Error loading bands for {record_id}: {e}")
            # Return zeros of correct shape
            return torch.zeros(
                (config.MODEL_INPUT_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
            ), torch.zeros((1, config.IMAGE_SIZE, config.IMAGE_SIZE))

        # ---------------------------------------------------------------------
        # 2. Construct 6-Channel Input
        # ---------------------------------------------------------------------

        # --- Part A: Ash False Color Composite (Channels 1-3) ---
        # Uses the labeled frame (IDX_CURRENT)

        # Red: T15 - T14
        r = normalize_range(
            b15[..., IDX_CURRENT] - b14[..., IDX_CURRENT], ASH_RED_MIN, ASH_RED_MAX
        )

        # Green: T14 - T11
        g = normalize_range(
            b14[..., IDX_CURRENT] - b11[..., IDX_CURRENT], ASH_GREEN_MIN, ASH_GREEN_MAX
        )

        # Blue: T14
        b = normalize_range(b14[..., IDX_CURRENT], ASH_BLUE_MIN, ASH_BLUE_MAX)

        ash_composite = np.stack([r, g, b], axis=-1)  # (H, W, 3)

        # --- Part B: Raw Temporal Differences (Channels 4-6) ---
        # Diff = Frame[t] - Frame[t-1]
        # We use raw values to preserve dynamic range of cooling physics

        diff_b11 = b11[..., IDX_CURRENT] - b11[..., IDX_PREV]
        diff_b14 = b14[..., IDX_CURRENT] - b14[..., IDX_PREV]
        diff_b15 = b15[..., IDX_CURRENT] - b15[..., IDX_PREV]

        # Stack differences (H, W, 3)
        diff_composite = np.stack([diff_b11, diff_b14, diff_b15], axis=-1)

        # --- Combine to 6 Channels ---
        # Shape: (H, W, 6)
        image = np.concatenate([ash_composite, diff_composite], axis=-1).astype(
            np.float32
        )

        # ---------------------------------------------------------------------
        # 3. Load Masks (Train/Val only)
        # ---------------------------------------------------------------------
        mask = None
        if self.stage in ["train", "validation"]:
            mask_path = os.path.join(config.INPUT_DIR, row["human_pixel_masks"])
            try:
                # Load mask: (H, W, 1)
                mask = np.load(mask_path).astype(np.float32)
            except Exception as e:
                print(f"Error loading mask for {record_id}: {e}")
                mask = np.zeros(
                    (config.IMAGE_SIZE, config.IMAGE_SIZE, 1), dtype=np.float32
                )

        # ---------------------------------------------------------------------
        # 4. Apply Augmentations
        # ---------------------------------------------------------------------
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

                # Albumentations ToTensorV2 converts HWC to CHW
                # Mask might come out as (H, W) or (1, H, W) depending on setup,
                # but ToTensorV2 usually handles image well.
                # For mask, we ensure it is (1, H, W)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.ndim == 3 and mask.shape[2] == 1:
                    # If it wasn't transposed by albumentations (rare for mask key), fix it
                    mask = mask.permute(2, 0, 1)
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # Return logic
        if self.stage in ["train", "validation"]:
            return image, mask
        else:
            return image, record_id


def get_dataloader(
    stage, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, debug=False
):
    """
    Factory function to create DataLoaders.

    Args:
        stage (str): 'train', 'validation', or 'test'.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        debug (bool): If True, subsets the dataset for quick debugging.
    """
    # Select metadata file
    if stage == "train":
        metadata_path = config.TRAIN_METADATA_PATH
        shuffle = True
    elif stage == "validation":
        metadata_path = config.VALIDATION_METADATA_PATH
        shuffle = False
    elif stage == "test":
        metadata_path = config.TEST_METADATA_PATH
        shuffle = False
    else:
        raise ValueError(f"Invalid stage: {stage}")

    # Get transforms
    transforms = get_transforms(stage)

    # Create Dataset
    dataset = ContrailsDataset(
        metadata_path=metadata_path, stage=stage, transform=transforms
    )

    # Debugging: Subset if requested
    if debug:
        subset_size = min(len(dataset), 100)
        indices = list(range(subset_size))
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"DEBUG MODE: Created {stage} dataloader with {subset_size} samples.")

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(stage == "train"),  # Drop last incomplete batch during training
    )

    return loader
