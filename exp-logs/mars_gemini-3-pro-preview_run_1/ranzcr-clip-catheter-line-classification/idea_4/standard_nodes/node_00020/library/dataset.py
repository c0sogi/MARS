import os
import cv2
import torch
import numpy as np
import pandas as pd
import ast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import rle_decode

# Prevent OpenCV from using multiple threads to avoid contention with PyTorch DataLoader
cv2.setNumThreads(0)


def load_annotations(load_cached_data=True):
    """
    Loads annotations from CSV or Parquet cache.
    Implements the required caching mechanism using Parquet to avoid pickle.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "cached_train_annotations.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # Fallback if cache is corrupt or unreadable
            pass

    # 2. Process from scratch if cache missing or load failed
    if os.path.exists(Config.ANNOTATIONS_PATH):
        df = pd.read_csv(Config.ANNOTATIONS_PATH)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # Save to cache
        df.to_parquet(cache_path)
        return df
    else:
        # Return empty dataframe if file missing (e.g. in some inference environments)
        return pd.DataFrame(columns=["StudyInstanceUID", "data"])


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the given mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                # CoarseDropout applied to image only to force learning from context.
                # Albumentations applies this to 'image' by default.
                A.CoarseDropout(
                    max_holes=8,
                    max_height=Config.IMAGE_SIZE // 20,
                    max_width=Config.IMAGE_SIZE // 20,
                    min_holes=1,
                    fill_value=0,
                    p=0.2,
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Val and Test
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )


class CatheterDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        mode="train",
        transform=None,
        load_cached_data=True,
        sample_size=None,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV (train/val/test).
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Transforms to apply.
            load_cached_data (bool): Whether to use cached annotations.
            sample_size (int, optional): If set, limits dataset size for debugging.
        """
        self.mode = mode
        self.transform = transform

        # Load Metadata
        self.df = pd.read_csv(metadata_path)

        if sample_size is not None:
            self.df = self.df.head(sample_size).reset_index(drop=True)

        # Pre-extract columns for faster access in __getitem__
        self.uids = self.df["StudyInstanceUID"].values
        self.file_paths = self.df["file_path"].values

        # Targets
        if self.mode != "test":
            self.targets = self.df[Config.TARGET_COLS].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load as RGB
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (should not happen with valid metadata)
            # Create a black image
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        uid = self.uids[idx]

        # 2. Augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # 3. Prepare Output
        output = {
            "image": image,
            "StudyInstanceUID": uid,
        }

        if self.targets is not None:
            output["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return output


def get_dataloader(
    metadata_path,
    mode="train",
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    sample_size=None,
):
    """
    Factory function to create dataloaders.
    """
    transform = get_transforms(mode=mode)

    dataset = CatheterDataset(
        metadata_path=metadata_path,
        mode=mode,
        transform=transform,
        load_cached_data=load_cached_data,
        sample_size=sample_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == "train"),
    )

    return dataloader
