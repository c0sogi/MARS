import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.utils import seed_everything

# Constants
INPUT_DIR = "./input"
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(mode="train", img_size=IMG_SIZE):
    """
    Creates the Albumentations transformation pipeline.

    Args:
        mode (str): 'train' or 'valid'.
        img_size (int): The size to resize images to.

    Returns:
        A.Compose: The transformation pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                # Moderate ColorJitter as per strategy
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )


class PawpularityDataset(Dataset):
    """
    Dataset class for the Pawpularity Contest.
    Loads images and metadata features.
    """

    def __init__(self, df, transforms=None, test=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing dataset metadata.
            transforms (A.Compose, optional): Albumentations transforms.
            test (bool): Whether this is the test set (returns Id instead of target).
        """
        self.df = df
        self.transforms = transforms
        self.test = test

        # Paths are relative to ./input in the metadata
        self.file_paths = [os.path.join(INPUT_DIR, fp) for fp in df["file_path"].values]
        self.ids = df["Id"].values

        # Identify feature columns (exclude non-feature columns)
        # Based on EDA, features are binary columns
        exclude_cols = ["Id", "file_path", "Pawpularity", "pawpularity_bins"]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        self.feature_cols.sort()  # Ensure deterministic order

        self.features = df[self.feature_cols].values.astype(np.float32)

        if not self.test:
            # Scale target to [0, 1] range
            self.targets = df["Pawpularity"].values.astype(np.float32) / 100.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load Image
        path = self.file_paths[idx]
        img = cv2.imread(path)

        if img is None:
            # Handle potential missing images gracefully
            # Create a black image of expected size
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]

        # Get Metadata
        meta = self.features[idx]

        if self.test:
            # Return Id for submission file generation
            return img, meta, self.ids[idx]
        else:
            # Return target for training/validation
            target = self.targets[idx]
            return img, meta, target
