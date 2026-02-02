import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config


def get_transforms(data, image_size):
    """
    Returns the Albumentations transformations for the specified data split and image size.

    Args:
        data (str): 'train', 'valid', or 'test'.
        image_size (int): Target resolution for resizing.

    Returns:
        A.Compose: The augmentation pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=Config.AUG_SHIFT_LIMIT,
                    scale_limit=Config.AUG_SCALE_LIMIT,
                    rotate_limit=Config.AUG_ROTATE_LIMIT,
                    p=Config.AUG_PROB,
                ),
                A.HorizontalFlip(p=0.5),
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data in ["valid", "test"]:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data split: {data}")


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transforms=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing file paths and labels.
            transforms (A.Compose): Albumentations transforms to apply.
        """
        self.df = df
        self.transforms = transforms
        self.file_paths = df["file_path"].values
        self.image_ids = df["image_id"].values

        # Check if target labels are present in the dataframe
        self.labels_present = all(col in df.columns for col in Config.CLASSES)
        if self.labels_present:
            self.labels = df[Config.CLASSES].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Construct full image path
        rel_path = self.file_paths[index]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            # Safety fallback for missing images (though metadata validation ensures existence)
            # Create a blank black image
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        result = {"image": image, "image_id": self.image_ids[index]}

        # Add targets if available
        if self.labels_present:
            target = self.labels[index]
            result["target"] = torch.tensor(target, dtype=torch.float32)
        else:
            # Return dummy target for test set to maintain consistency
            result["target"] = torch.tensor(
                np.zeros(Config.NUM_CLASSES), dtype=torch.float32
            )

        return result


def load_dataset_dfs(load_cached_data=True):
    """
    Loads train, val, and test dataframes.
    Implements caching mechanism using parquet files in the working directory.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_df.parquet")
    val_cache = os.path.join(cache_dir, "val_df.parquet")
    test_cache = os.path.join(cache_dir, "test_df.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception:
                # If loading fails, proceed to recompute/reload
                pass

    # 2. Load from source metadata CSVs
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 3. Save to cache
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df
