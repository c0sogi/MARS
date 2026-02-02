import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import extract_file_sizes


def get_transforms(split="train"):
    """
    Returns the image transformations for the specified split.
    Uses dataset-specific mean and std derived from analysis.
    """
    # Stats from data analysis
    # Mean: R=128.37, G=115.25, B=119.40 -> [0.5034, 0.4520, 0.4682]
    # Std:  R=38.60,  G=35.68,  B=39.15  -> [0.1514, 0.1399, 0.1535]
    mean = (0.5034, 0.4520, 0.4682)
    std = (0.1514, 0.1399, 0.1535)

    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])


class CactusDataset(Dataset):
    """
    Dataset class that loads all images into RAM for fast training.
    Implements caching of decoded images to .npy files.
    """

    def __init__(
        self,
        df,
        data_dir,
        transform=None,
        cache_name="imgs.npy",
        load_cached_data=True,
        aux_targets=None,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            data_dir (str): Root directory containing images (relative paths in df).
            transform (albumentations.Compose): Transforms to apply.
            cache_name (str): Filename for caching the image array.
            load_cached_data (bool): Whether to use the cache.
            aux_targets (np.array): Pre-computed auxiliary targets (normalized file sizes).
        """
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.aux_targets = aux_targets

        # Load images (with caching logic)
        self.images = self._load_images(cache_name, load_cached_data)

        # Load labels
        if "has_cactus" in df.columns:
            self.labels = df["has_cactus"].values.astype(np.float32)
        else:
            # Placeholder for test set if column missing (though metadata usually has it)
            self.labels = np.zeros(len(df), dtype=np.float32)

    def _load_images(self, cache_name, load_cached_data):
        """
        Loads images from cache or reads from disk and caches them.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(Config.CACHE_DIR, cache_name)

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                imgs = np.load(cache_path)
                # print(f"Loaded images from cache: {cache_path}")
                return imgs
            except Exception as e:
                print(f"Failed to load image cache {cache_path}: {e}")

        # 2. Compute (Read Images)
        # print(f"Reading images from {self.data_dir}...")
        imgs = []
        # Use file_path from metadata
        paths = self.df["file_path"].values

        for rel_path in paths:
            full_path = os.path.join(self.data_dir, rel_path)
            img = cv2.imread(full_path)

            if img is None:
                # Fallback for missing files (should not happen with validated metadata)
                img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            imgs.append(img)

        imgs = np.array(imgs, dtype=np.uint8)

        # 3. Save Cache
        np.save(cache_path, imgs)
        # print(f"Saved images to cache: {cache_path}")

        return imgs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve image from RAM
        image = self.images[idx]

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Retrieve label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Retrieve auxiliary target
        if self.aux_targets is not None:
            aux = torch.tensor(self.aux_targets[idx], dtype=torch.float32)
        else:
            aux = torch.tensor(0.0, dtype=torch.float32)

        return image, label, aux


def get_datasets(load_cached_data=True):
    """
    Factory function to create Train, Validation, and Test datasets.
    Handles metadata loading and auxiliary target computation.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)
    test_df = pd.read_csv(Config.TEST_META_PATH)

    # 2. Compute Auxiliary Targets (File Sizes)
    # Train
    train_aux, stats = extract_file_sizes(
        train_df,
        Config.INPUT_DIR,
        "cache_train_fsizes.npy",
        load_cached_data=load_cached_data,
        normalization_stats=None,
    )

    # Val (use train stats)
    val_aux, _ = extract_file_sizes(
        val_df,
        Config.INPUT_DIR,
        "cache_val_fsizes.npy",
        load_cached_data=load_cached_data,
        normalization_stats=stats,
    )

    # Test (use train stats)
    test_aux, _ = extract_file_sizes(
        test_df,
        Config.INPUT_DIR,
        "cache_test_fsizes.npy",
        load_cached_data=load_cached_data,
        normalization_stats=stats,
    )

    # 3. Create Datasets
    train_ds = CactusDataset(
        df=train_df,
        data_dir=Config.INPUT_DIR,
        transform=get_transforms("train"),
        cache_name="cache_train_imgs.npy",
        load_cached_data=load_cached_data,
        aux_targets=train_aux,
    )

    val_ds = CactusDataset(
        df=val_df,
        data_dir=Config.INPUT_DIR,
        transform=get_transforms("val"),
        cache_name="cache_val_imgs.npy",
        load_cached_data=load_cached_data,
        aux_targets=val_aux,
    )

    test_ds = CactusDataset(
        df=test_df,
        data_dir=Config.INPUT_DIR,
        transform=get_transforms("test"),
        cache_name="cache_test_imgs.npy",
        load_cached_data=load_cached_data,
        aux_targets=test_aux,
    )

    return train_ds, val_ds, test_ds
