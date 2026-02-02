import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from library.config import CFG


def get_transforms(data="train", image_size=256):
    """
    Generates the Albumentations transformation pipeline.

    Args:
        data (str): Type of data ('train', 'valid', or 'test').
        image_size (int): Target resolution for resizing.

    Returns:
        A.Compose: The composition of transformations.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.Normalize(
                    mean=CFG.mean,
                    std=CFG.std,
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=CFG.mean,
                    std=CFG.std,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Fallback to validation transforms
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=CFG.mean,
                    std=CFG.std,
                ),
                ToTensorV2(),
            ]
        )


class WhaleDataset(Dataset):
    def __init__(
        self, df, transform=None, root_dir=None, label_col="label_idx", id_col="Id"
    ):
        """
        Dataset class for Whale Images.

        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_path, Id, etc.).
            transform (albumentations.Compose): Image transformations.
            root_dir (str): Root directory where images are stored (usually CFG.input_root).
            label_col (str): Column name for the integer-encoded label.
            id_col (str): Column name for the string Whale ID.
        """
        self.df = df
        self.transform = transform
        self.root_dir = root_dir if root_dir else CFG.input_root
        self.label_col = label_col
        self.id_col = id_col

        # Check availability of columns
        self.has_label = self.label_col in df.columns
        self.has_id = self.id_col in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Metadata 'file_path' is relative (e.g., "train/img.jpg")
        # We join it with the root directory.
        image_path = os.path.join(self.root_dir, row["file_path"])

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            # Safety fallback for missing/corrupt images to prevent training crash
            # Create a blank black image
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform to tensor if no pipeline provided
            t = ToTensorV2()
            image = t(image=image)["image"]

        result = {"image": image, "image_name": row["Image"]}

        # Add label if available (for training/validation)
        if self.has_label:
            result["label"] = torch.tensor(row[self.label_col], dtype=torch.long)

        # Add string ID if available (for tracking/inference)
        if self.has_id:
            result["id"] = row[self.id_col]

        return result


def process_data(load_cached_data=True):
    """
    Loads metadata, filters datasets according to the strategy, encodes labels,
    and caches the processed DataFrames.

    Strategy Compliance:
    - Excludes 'new_whale' from Training and Validation sets to support
      Closed-Set Metric Learning.
    - Caches processed data to parquet/npy to avoid re-processing.

    Args:
        load_cached_data (bool): If True, attempts to load from disk cache.

    Returns:
        tuple: (train_df, val_df, test_df, num_classes)
    """
    cache_dir = CFG.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")
    encoder_cache = os.path.join(cache_dir, "label_encoder_classes.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
            and os.path.exists(encoder_cache)
        ):
            print(f"Loading cached data from {cache_dir}...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            classes = np.load(encoder_cache, allow_pickle=True)
            num_classes = len(classes)
            return train_df, val_df, test_df, num_classes

    # 2. Process from Scratch
    print("Processing data from metadata...")

    # Load raw metadata
    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.val_csv)
    test_df = pd.read_csv(CFG.test_csv)

    # Filter 'new_whale' from training data
    # Strategy: "The training set will strictly exclude the new_whale class"
    train_df = train_df[train_df["Id"] != "new_whale"].reset_index(drop=True)

    # Filter 'new_whale' from validation data
    # Strategy: "Validation Gallery... new_whale class will be excluded"
    val_df = val_df[val_df["Id"] != "new_whale"].reset_index(drop=True)

    # Fit Label Encoder on Known Classes
    le = LabelEncoder()
    le.fit(train_df["Id"])

    # Transform Labels
    train_df["label_idx"] = le.transform(train_df["Id"])

    # Ensure Validation set only contains known classes (subset check)
    # (Metadata split logic + new_whale filtering should guarantee this, but we filter to be safe)
    val_df = val_df[val_df["Id"].isin(le.classes_)].reset_index(drop=True)
    val_df["label_idx"] = le.transform(val_df["Id"])

    # Save to Cache
    print(f"Saving processed data to {cache_dir}...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)
    np.save(encoder_cache, le.classes_)

    num_classes = len(le.classes_)

    print(f"Data Processing Complete.")
    print(f"  Train Samples: {len(train_df)}")
    print(f"  Val Samples:   {len(val_df)}")
    print(f"  Test Samples:  {len(test_df)}")
    print(f"  Num Classes:   {num_classes}")

    return train_df, val_df, test_df, num_classes
