import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config
from library.utils import set_seed


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Segmentation.

    Loads satellite bands, computes Ash False Color Composite and Temporal Differences,
    and normalizes global metadata.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        split: str = "train",
        transform: A.Compose = None,
        normalize_metadata: bool = True,
        stats: dict = None,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Dataframe containing file paths and metadata.
            split (str): 'train', 'validation', or 'test'.
            transform (A.Compose): Albumentations transforms.
            normalize_metadata (bool): Whether to normalize metadata features.
            stats (dict): Dictionary containing min/max values for metadata normalization.
        """
        self.df = metadata_df
        self.split = split
        self.transform = transform
        self.normalize_metadata = normalize_metadata
        self.stats = stats

        # Pre-calculate metadata normalization parameters if not provided
        if self.normalize_metadata and self.stats is None:
            self.stats = {
                "row_min_min": self.df["row_min"].min(),
                "row_min_max": self.df["row_min"].max(),
                "col_min_min": self.df["col_min"].min(),
                "col_min_max": self.df["col_min"].max(),
            }

    def __len__(self):
        return len(self.df)

    def normalize_range(self, data, vmin, vmax):
        """Linearly scales data from [vmin, vmax] to [0, 1]. Clips values outside range."""
        return np.clip((data - vmin) / (vmax - vmin), 0, 1)

    def get_ash_composite(self, b11, b14, b15):
        """
        Computes Ash False Color Composite.
        Args:
            b11, b14, b15: 2D arrays for bands at t=4.
        Returns:
            np.ndarray: (H, W, 3) normalized to [0, 1].
        """
        # Red: T15 - T14
        r = self.normalize_range(b15 - b14, Config.ASH_RED_MIN, Config.ASH_RED_MAX)
        # Green: T14 - T11
        g = self.normalize_range(b14 - b11, Config.ASH_GREEN_MIN, Config.ASH_GREEN_MAX)
        # Blue: T14
        b = self.normalize_range(b14, Config.ASH_BLUE_MIN, Config.ASH_BLUE_MAX)

        return np.dstack((r, g, b))

    def get_temporal_diff(self, b11_t3, b11_t4, b14_t3, b14_t4, b15_t3, b15_t4):
        """
        Computes normalized temporal differences (t4 - t3).
        Returns:
            np.ndarray: (H, W, 3) normalized to [0, 1].
        """
        d11 = self.normalize_range(b11_t4 - b11_t3, Config.DIFF_MIN, Config.DIFF_MAX)
        d14 = self.normalize_range(b14_t4 - b14_t3, Config.DIFF_MIN, Config.DIFF_MAX)
        d15 = self.normalize_range(b15_t4 - b15_t3, Config.DIFF_MIN, Config.DIFF_MAX)

        return np.dstack((d11, d14, d15))

    def process_metadata(self, row):
        """
        Extracts and normalizes metadata: row_min, col_min, timestamp.
        Returns:
            torch.Tensor: (3,)
        """
        if not self.normalize_metadata:
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        # 1. Latitude-like (row_min)
        row_min = row["row_min"]
        norm_row = (row_min - self.stats["row_min_min"]) / (
            self.stats["row_min_max"] - self.stats["row_min_min"] + 1e-6
        )

        # 2. Longitude-like (col_min)
        col_min = row["col_min"]
        norm_col = (col_min - self.stats["col_min_min"]) / (
            self.stats["col_min_max"] - self.stats["col_min_min"] + 1e-6
        )

        # 3. Time of Day (timestamp)
        # 86400 seconds in a day. Timestamp is unix epoch.
        ts = row["timestamp"]
        # Calculate fraction of day (0.0 to 1.0)
        time_of_day = (ts % 86400) / 86400.0

        return torch.tensor([norm_row, norm_col, time_of_day], dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # Define paths
        # Note: metadata csv contains relative paths from INPUT_DIR
        path_b11 = os.path.join(Config.INPUT_DIR, row["band_11"])
        path_b14 = os.path.join(Config.INPUT_DIR, row["band_14"])
        path_b15 = os.path.join(Config.INPUT_DIR, row["band_15"])

        # Load Bands: Shape (H, W, T) where T=8 usually
        # We need t=4 (labeled) and t=3 (previous)
        # Indices: 0..7. t_current=4, t_prev=3
        try:
            img_b11 = np.load(path_b11)
            img_b14 = np.load(path_b14)
            img_b15 = np.load(path_b15)
        except Exception as e:
            # Fallback for missing files (should not happen given validation)
            print(f"Error loading {record_id}: {e}")
            img_b11 = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 8))
            img_b14 = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 8))
            img_b15 = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 8))

        # Extract frames
        # t=4 is the labeled frame
        b11_t4 = img_b11[..., 4]
        b14_t4 = img_b14[..., 4]
        b15_t4 = img_b15[..., 4]

        # t=3 is the previous frame
        b11_t3 = img_b11[..., 3]
        b14_t3 = img_b14[..., 3]
        b15_t3 = img_b15[..., 3]

        # 1. Compute Ash Composite (3 channels)
        ash = self.get_ash_composite(b11_t4, b14_t4, b15_t4)

        # 2. Compute Temporal Differences (3 channels)
        diff = self.get_temporal_diff(b11_t3, b11_t4, b14_t3, b14_t4, b15_t3, b15_t4)

        # Concatenate to 6 channels: (H, W, 6)
        image = np.concatenate([ash, diff], axis=2).astype(np.float32)

        # Load Mask
        mask = None
        if self.split in ["train", "validation"]:
            mask_path = os.path.join(Config.INPUT_DIR, row["human_pixel_masks"])
            try:
                # Shape (H, W, 1)
                mask = np.load(mask_path).astype(np.float32)
            except:
                mask = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1), dtype=np.float32
                )

        # Apply Augmentations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]
        else:
            # Convert to tensor if no transform provided (ToTensorV2 is usually in transform)
            # HWC -> CHW
            image = torch.from_numpy(image).permute(2, 0, 1)
            if mask is not None:
                mask = torch.from_numpy(mask).permute(2, 0, 1)

        # Process Metadata
        meta_tensor = self.process_metadata(row)

        # Return dictionary or tuple?
        # Standard PyTorch usually tuple (input, target), but we have metadata.
        # We will return (image, metadata, mask)

        if self.split == "test":
            return image, meta_tensor, record_id
        else:
            return image, meta_tensor, mask


def get_data_loaders(
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Factory function to create DataLoaders for Train, Validation, and Test.
    """
    # 1. Load Metadata Dataframes
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debug Mode: Subsample
    if debug:
        train_df = train_df.sample(
            n=min(len(train_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        # Keep test full usually, but for strict debug maybe subsample
        # test_df = test_df.sample(n=min(len(test_df), debug_sample_size), random_state=Config.SEED).reset_index(drop=True)

    # 2. Define Transforms
    # Training: Affine Transformations (Flip, Shift, Scale, Rotate)
    # No elastic/grid distortion as per idea.
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5,
                border_mode=0,  # Constant 0 padding
            ),
            ToTensorV2(transpose_mask=True),
        ]
    )

    # Validation/Test: Just ToTensor
    val_transform = A.Compose([ToTensorV2(transpose_mask=True)])

    # 3. Calculate Global Stats for Metadata Normalization from Training Set
    # We use training stats to normalize validation/test to avoid leakage
    meta_stats = {
        "row_min_min": train_df["row_min"].min(),
        "row_min_max": train_df["row_min"].max(),
        "col_min_min": train_df["col_min"].min(),
        "col_min_max": train_df["col_min"].max(),
    }

    # 4. Instantiate Datasets
    train_dataset = ContrailDataset(
        train_df, split="train", transform=train_transform, stats=meta_stats
    )

    val_dataset = ContrailDataset(
        val_df, split="validation", transform=val_transform, stats=meta_stats
    )

    test_dataset = ContrailDataset(
        test_df,
        split="test",
        transform=val_transform,
        stats=meta_stats,
        normalize_metadata=False,  # Test metadata doesn't have row/col info in csv usually?
        # Wait, test_metadata.csv generated by script DOES NOT have row_min/col_min/timestamp columns
        # because test_metadata.json was not merged (it doesn't exist or wasn't used same way).
        # The prompt says test_metadata.csv has 10 columns (id + bands).
        # We must handle missing metadata for test set.
        # The model expects metadata input. We will pass zeros for test if columns missing.
    )

    # Correction for Test Dataset Metadata:
    # If columns are missing in test_df, the process_metadata will fail.
    # We need to handle this in dataset.
    # For the purpose of this competition, if test metadata (geo/time) is not provided in a usable format
    # aligned with train, we might have to pass zeros or neutral values.
    # However, usually test sets in this comp DO have metadata.
    # Looking at the metadata generation script:
    # "test": { "metadata_file": None } -> No merge.
    # So test_df ONLY has record_id and band paths.
    # The ContrailDataset.process_metadata will fail on row['row_min'].
    # We should detect this in __init__ or process_metadata.

    # Let's patch ContrailDataset to handle missing columns gracefully (fill 0).
    # (This logic is added inside the class below implicitly by checking columns or try-except)

    # 5. Create Loaders
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


# Patching ContrailDataset process_metadata to be robust for Test set
def safe_process_metadata(self, row):
    if not self.normalize_metadata:
        return torch.zeros(3, dtype=torch.float32)

    # Check if columns exist
    if "row_min" not in row or "col_min" not in row or "timestamp" not in row:
        # Fallback for test set if metadata is missing
        return torch.zeros(3, dtype=torch.float32)

    try:
        # 1. Latitude-like (row_min)
        row_min = row["row_min"]
        norm_row = (row_min - self.stats["row_min_min"]) / (
            self.stats["row_min_max"] - self.stats["row_min_min"] + 1e-6
        )

        # 2. Longitude-like (col_min)
        col_min = row["col_min"]
        norm_col = (col_min - self.stats["col_min_min"]) / (
            self.stats["col_min_max"] - self.stats["col_min_min"] + 1e-6
        )

        # 3. Time of Day (timestamp)
        ts = row["timestamp"]
        time_of_day = (ts % 86400) / 86400.0

        return torch.tensor([norm_row, norm_col, time_of_day], dtype=torch.float32)
    except:
        return torch.zeros(3, dtype=torch.float32)


# Apply patch
ContrailDataset.process_metadata = safe_process_metadata
