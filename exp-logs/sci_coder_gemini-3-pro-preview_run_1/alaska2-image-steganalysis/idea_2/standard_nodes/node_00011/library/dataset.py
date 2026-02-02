import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import seed_everything


def get_transforms(split):
    """
    Returns the Albumentations transform pipeline based on the dataset split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if split == "train":
        # Dihedral Group D4 Augmentations:
        # Combinations of Horizontal/Vertical flips and 90-degree rotations
        # cover the 8 symmetries of the square.
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        # For Validation and Test, we only convert to Tensor.
        # No resizing is performed as per the requirement to preserve stego artifacts.
        return A.Compose([ToTensorV2()])


def load_metadata(split, load_cached_data=True):
    """
    Loads metadata for the specified split, handling caching and debug subsampling.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    # Ensure working directory exists for cache
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORK_DIR, f"{split}_metadata.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Verify if debug mode matches the cached data size roughly
            # (Simple heuristic: if debug is on, size should be small)
            if Config.DEBUG and len(df) > Config.DEBUG_SAMPLE_SIZE:
                # Cache is full data but we want debug; reload from source
                pass
            else:
                return df
        except Exception:
            # If load fails, fall back to processing from scratch
            pass

    # 2. Compute/Process data from scratch
    if split == "train":
        csv_path = Config.TRAIN_CSV
    elif split == "val":
        csv_path = Config.VAL_CSV
    elif split == "test":
        csv_path = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Handle Debug Mode
    if Config.DEBUG:
        if split == "train" or split == "val":
            # Stratified sample if possible, else random
            if "label" in df.columns:
                # Sample min(len, DEBUG_SAMPLE_SIZE)
                n_sample = min(len(df), Config.DEBUG_SAMPLE_SIZE)
                df = df.sample(n=n_sample, random_state=Config.SEED).reset_index(
                    drop=True
                )
            else:
                df = df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        elif split == "test":
            df = df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

    return df


class StegoDataset(Dataset):
    """
    Dataset class for Steganography Detection.
    """

    def __init__(self, split, load_cached_data=True, transform=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached metadata.
            transform (A.Compose, optional): Custom transforms. If None, defaults are used.
        """
        self.split = split
        self.df = load_metadata(split, load_cached_data=load_cached_data)
        self.root_dir = Config.INPUT_ROOT

        if transform is None:
            self.transform = get_transforms(split)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # row['file_path'] is relative to input root (e.g., "Cover/00001.jpg")
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image
        # cv2 loads in BGR, convert to RGB
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocessing:
        # Convert to float32 and scale to [0, 1].
        # We do not normalize with ImageNet mean/std because the model uses a
        # custom Residual Extraction Bank that expects raw pixel statistics.
        image = image.astype(np.float32) / 255.0

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return based on split
        if self.split == "test":
            # For test, we need the image_id for submission
            return image, row["image_id"]
        else:
            # For train/val, we return the label
            # Label is converted to float for BCEWithLogitsLoss
            label = torch.tensor(row["label"], dtype=torch.float32)
            return image, label
