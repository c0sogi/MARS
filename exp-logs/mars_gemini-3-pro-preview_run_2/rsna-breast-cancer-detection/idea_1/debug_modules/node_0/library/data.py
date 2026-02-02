import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    CACHE_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import read_dicom_from_bytes

# Define feature columns for tabular data
CATEGORICAL_COLS = ["site_id", "laterality", "view", "implant", "machine_id"]
NUMERICAL_COLS = ["age"]
TARGET_COL = "cancer"
ID_COLS = ["patient_id", "image_id", "prediction_id", "file_path", "split"]


def get_transforms(phase: str):
    """
    Returns Albumentations transforms for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def process_and_cache_metadata(load_cached_data: bool = True):
    """
    Loads metadata, performs tabular preprocessing (imputation, normalization, OHE),
    and caches the result to disk.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "processed_metadata.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached metadata from {cache_path}")
        df_all = pd.read_parquet(cache_path)
    else:
        print("Processing metadata from scratch...")
        # Load raw metadata
        train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
        val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
        test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

        # Mark splits
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

        # Concatenate to ensure consistent encoding
        # Note: Test set does not have 'cancer' or other train-only cols.
        # We align columns by keeping common ones + target for train/val.
        common_cols = list(set(train_df.columns) & set(test_df.columns))
        # Ensure we keep the target and split info
        cols_to_keep = list(
            set(
                common_cols + CATEGORICAL_COLS + NUMERICAL_COLS + ID_COLS + [TARGET_COL]
            )
        )

        # Filter columns to relevant ones to avoid mismatch issues during concat
        train_subset = train_df[train_df.columns.intersection(cols_to_keep)]
        val_subset = val_df[val_df.columns.intersection(cols_to_keep)]
        test_subset = test_df[test_df.columns.intersection(cols_to_keep)]

        df_all = pd.concat(
            [train_subset, val_subset, test_subset],
            axis=0,
            ignore_index=True,
            sort=False,
        )

        # 1. Handle Numerical: Age
        # Impute missing age with mean
        age_mean = df_all["age"].mean()
        df_all["age"] = df_all["age"].fillna(age_mean)
        # Normalize Age
        age_std = df_all["age"].std()
        df_all["age"] = (df_all["age"] - age_mean) / (age_std + 1e-7)

        # 2. Handle Categorical: OHE
        # Convert machine_id to string to treat as categorical
        if "machine_id" in df_all.columns:
            df_all["machine_id"] = df_all["machine_id"].astype(str)

        # One-Hot Encoding
        df_all = pd.get_dummies(
            df_all, columns=CATEGORICAL_COLS, dummy_na=False, dtype=float
        )

        # Save to cache
        print(f"Saving processed metadata to {cache_path}")
        df_all.to_parquet(cache_path)

    # Split back
    train_df = df_all[df_all["split"] == "train"].reset_index(drop=True)
    val_df = df_all[df_all["split"] == "val"].reset_index(drop=True)
    test_df = df_all[df_all["split"] == "test"].reset_index(drop=True)

    # Identify feature columns (all columns that are not IDs or Target)
    # We exclude the original ID columns and the target
    exclude_cols = set(ID_COLS + [TARGET_COL])
    feature_cols = [c for c in df_all.columns if c not in exclude_cols]

    return train_df, val_df, test_df, feature_cols


class BreastCancerDataset(Dataset):
    def __init__(self, df, feature_cols, transforms=None, mode="train"):
        self.df = df
        self.feature_cols = feature_cols
        self.transforms = transforms
        self.mode = mode

        # Pre-extract tabular data as float32 numpy array for speed
        self.tabular_data = self.df[self.feature_cols].values.astype(np.float32)

        # Pre-extract targets if available
        if self.mode in ["train", "val"]:
            self.targets = self.df[TARGET_COL].values.astype(np.float32)
        else:
            self.targets = None

        # Pre-extract file paths and prediction IDs
        self.file_paths = self.df["file_path"].values
        if self.mode == "test":
            self.prediction_ids = self.df["prediction_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        rel_path = self.file_paths[idx]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Use the utility function to read bytes and decode
        image = read_dicom_from_bytes(full_path)

        # 2. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # 3. Get Tabular Features
        tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float32)

        # 4. Return based on mode
        if self.mode in ["train", "val"]:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return (image, tabular), target
        else:
            pred_id = self.prediction_ids[idx]
            # Return a dict or tuple. Dict is safer for mixed types in some loops,
            # but tuple (inputs, meta) is standard.
            # Here we return inputs tuple and prediction_id
            return (image, tabular), pred_id


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Prepares and returns DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached metadata.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        train_loader, val_loader, test_loader, num_tabular_features
    """
    # 1. Process Metadata
    train_df, val_df, test_df, feature_cols = process_and_cache_metadata(
        load_cached_data
    )

    if debug:
        print("Debug mode: Subsetting data...")
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    print(
        f"Data sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    print(f"Number of tabular features: {len(feature_cols)}")

    # 2. Create Datasets
    train_dataset = BreastCancerDataset(
        train_df, feature_cols, transforms=get_transforms("train"), mode="train"
    )

    val_dataset = BreastCancerDataset(
        val_df, feature_cols, transforms=get_transforms("val"), mode="val"
    )

    test_dataset = BreastCancerDataset(
        test_df, feature_cols, transforms=get_transforms("test"), mode="test"
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, len(feature_cols)
