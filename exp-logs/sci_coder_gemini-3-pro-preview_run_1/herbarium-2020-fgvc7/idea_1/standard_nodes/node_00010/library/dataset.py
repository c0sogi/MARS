import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from pathlib import Path

from library.config import (
    INPUT_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    CACHE_DIR,
    IMG_SIZE,
    DEBUG_SAMPLE_SIZE,
    SEED,
)


def get_label_mapping(load_cached_data=True):
    """
    Generates or loads a mapping from category_id to label_idx (0..N-1).
    Caching mechanism implemented using Parquet.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (label2id, id2label) dictionaries.
    """
    cache_path = CACHE_DIR / "label_mapping.parquet"

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Try to load cached data
    if load_cached_data and cache_path.exists():
        try:
            mapping_df = pd.read_parquet(cache_path)
            label2id = dict(zip(mapping_df["category_id"], mapping_df["label_idx"]))
            id2label = dict(zip(mapping_df["label_idx"], mapping_df["category_id"]))
            return label2id, id2label
        except Exception:
            pass  # Fallback to recomputing if load fails

    # 2. Compute from scratch
    if not TRAIN_META_PATH.exists():
        raise FileNotFoundError(f"Training metadata not found at {TRAIN_META_PATH}")

    df = pd.read_csv(TRAIN_META_PATH)
    unique_cats = sorted(df["category_id"].unique())

    mapping_data = {
        "category_id": unique_cats,
        "label_idx": list(range(len(unique_cats))),
    }
    mapping_df = pd.DataFrame(mapping_data)

    # Save to cache
    mapping_df.to_parquet(cache_path, index=False)

    label2id = dict(zip(mapping_df["category_id"], mapping_df["label_idx"]))
    id2label = dict(zip(mapping_df["label_idx"], mapping_df["category_id"]))

    return label2id, id2label


def get_transforms(split):
    """
    Returns albumentations transforms for the given split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    # Standard ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if split == "train":
        return A.Compose(
            [
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # val or test
        return A.Compose(
            [
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class PlantDataset(Dataset):
    def __init__(self, split, transform=None, debug=False):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Transforms to apply.
            debug (bool): If True, subsample the dataset for debugging.
        """
        self.split = split
        self.transform = transform
        self.debug = debug

        # Select Metadata File
        if split == "train":
            self.meta_path = TRAIN_META_PATH
        elif split == "val":
            self.meta_path = VAL_META_PATH
        elif split == "test":
            self.meta_path = TEST_META_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load Metadata
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.meta_path}")

        self.df = pd.read_csv(self.meta_path)

        # Debug Subsampling
        if self.debug:
            self.df = self.df.sample(
                n=min(len(self.df), DEBUG_SAMPLE_SIZE), random_state=SEED
            ).reset_index(drop=True)

        # Load Label Mapping for Train/Val
        if self.split in ["train", "val"]:
            self.label2id, _ = get_label_mapping(load_cached_data=True)

            # Filter out any rows with categories not in the mapping
            valid_cats = set(self.label2id.keys())
            self.df = self.df[self.df["category_id"].isin(valid_cats)].reset_index(
                drop=True
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct Path
        # file_path in csv is relative to input/ e.g. "nybg2020/train/..."
        img_path = str(INPUT_DIR / row["file_path"])

        # Load Image
        image = cv2.imread(img_path)

        # Handle potential read errors
        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = ToTensorV2()(image=image)["image"]

        # Return based on split
        if self.split == "test":
            # Return image and image_id for inference mapping
            return image, row["image_id"]
        else:
            # Return image and label index
            cat_id = row["category_id"]
            label = self.label2id[cat_id]
            return image, torch.tensor(label, dtype=torch.long)
