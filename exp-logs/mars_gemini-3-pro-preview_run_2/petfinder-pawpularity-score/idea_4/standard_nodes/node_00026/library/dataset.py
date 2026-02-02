import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import scale_target


def load_metadata_splits(
    debug: bool = Config.debug, sample_size: int = Config.debug_sample_size
):
    """
    Loads the train, validation, and test metadata DataFrames.
    Applies debugging sampling if requested.

    Args:
        debug (bool): If True, limits the dataframe size for debugging.
        sample_size (int): Number of samples to load in debug mode.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Load DataFrames from the pre-generated metadata paths
    train_df = pd.read_csv(Config.train_metadata_path)
    val_df = pd.read_csv(Config.val_metadata_path)
    test_df = pd.read_csv(Config.test_metadata_path)

    if debug:
        print(f"Debug mode enabled. Subsampling to {sample_size} rows.")
        train_df = train_df.head(sample_size)
        val_df = val_df.head(sample_size)
        test_df = test_df.head(sample_size)

    return train_df, val_df, test_df


def get_transforms(data: str):
    """
    Creates the Albumentations transformation pipeline.

    Args:
        data (str): The mode of operation ('train', 'valid', 'test').

    Returns:
        A.Compose: The composition of transforms.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.image_size, Config.image_size),
                A.HorizontalFlip(p=0.5),
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
                A.Resize(Config.image_size, Config.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class PawpularityDataset(Dataset):
    """
    PyTorch Dataset for the Pet Pawpularity Prediction task.
    """

    def __init__(self, df: pd.DataFrame, transforms=None, test: bool = False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and targets.
            transforms (A.Compose, optional): Albumentations transforms.
            test (bool): If True, treats data as test set (no target expected).
        """
        self.df = df
        self.transforms = transforms
        self.test = test
        self.root_dir = Config.input_dir

        # Pre-fetch file paths to avoid DataFrame overhead in __getitem__
        self.file_paths = df["file_path"].values

        # Pre-fetch metadata (Cite solution_lesson_node_00007)
        self.meta_features = df[Config.meta_cols].values.astype(np.float32)

        if not self.test:
            # Ensure target column exists for training/validation
            if Config.target_col not in df.columns:
                raise ValueError(
                    f"Target column '{Config.target_col}' missing in DataFrame."
                )
            self.targets = df[Config.target_col].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Construct absolute file path
        # Metadata file_path is relative to input directory (e.g., "train/xyz.jpg")
        img_path = os.path.join(self.root_dir, self.file_paths[index])

        # Load image via OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Get metadata
        meta = torch.tensor(self.meta_features[index], dtype=torch.float32)

        # Handle target
        if self.test:
            # Return dummy target for test set
            target = torch.tensor(0.0, dtype=torch.float32)
        else:
            # Get raw target and scale it to [0, 1]
            raw_target = self.targets[index]
            scaled_val = scale_target(raw_target)
            target = torch.tensor(scaled_val, dtype=torch.float32)

        return image, meta, target
