import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the albumentations transform pipeline for the given mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test transforms (Deterministic)
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class ArtworkDataset(Dataset):
    def __init__(
        self, mode="train", load_cached_data=True, transform=None, data_limit=None
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load processed metadata from cache.
            transform (A.Compose): Albumentations transforms.
            data_limit (int, optional): Limit dataset size for debugging/testing.
        """
        self.mode = mode
        self.transform = transform
        self.input_dir = Config.INPUT_DIR
        self.num_classes = Config.NUM_CLASSES

        # Ensure working directory exists for caching
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"{mode}_processed.parquet")

        # Load Data
        self.df = self._load_data(load_cached_data)

        # Apply data limit if specified (for debugging)
        if data_limit is not None:
            self.df = self.df.iloc[:data_limit]

    def _load_data(self, load_cached_data):
        """
        Loads metadata from cache or processes raw CSVs.
        """
        # 1. Try to load from cache
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                df = pd.read_parquet(self.cache_path)
                return df
            except Exception:
                # If load fails, fall through to compute from scratch
                pass

        # 2. Compute from scratch
        if self.mode == "train":
            meta_path = Config.TRAIN_METADATA
        elif self.mode == "val":
            meta_path = Config.VAL_METADATA
        else:
            meta_path = Config.TEST_METADATA

        # Load CSV with string types to preserve IDs and label lists
        df = pd.read_csv(meta_path, dtype={"id": str, "attribute_ids": str})

        # Handle NaN in attribute_ids for train/val
        if "attribute_ids" in df.columns:
            df["attribute_ids"] = df["attribute_ids"].fillna("")

        # Save to cache for future runs
        df.to_parquet(self.cache_path, index=False)

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # file_path in metadata is relative (e.g., "train/xxx.png")
        rel_path = row["file_path"]
        full_path = os.path.join(self.input_dir, rel_path)

        # Read Image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for potentially corrupt/missing images
            # Create a blank black image to prevent crashing
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Default transform if none provided
            t = ToTensorV2()
            image = t(image=image)["image"]

        # Return logic based on mode
        if self.mode in ["train", "val"]:
            # Create Multi-hot Target
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            attr_str = row.get("attribute_ids", "")

            if attr_str and isinstance(attr_str, str) and attr_str.strip():
                # Parse space-separated integers
                ids = [int(x) for x in attr_str.split()]
                target[ids] = 1.0

            return image, target
        else:
            # Test mode: return image and id for submission mapping
            return image, row["id"]
