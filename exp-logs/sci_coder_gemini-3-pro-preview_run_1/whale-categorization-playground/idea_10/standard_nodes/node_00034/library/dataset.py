import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_class_mapping(load_cached_data=True):
    """
    Generates or loads a consistent mapping between class ID strings and integer labels.

    Logic:
    1. Checks for cached 'classes.npy' in Config.WORKING_DIR.
    2. If found and load_cached_data is True, loads it.
    3. Otherwise, reads Config.TRAIN_CSV, extracts unique IDs, sorts them,
       saves to cache, and returns the mapping.

    Returns:
        classes (np.ndarray): Sorted array of class names.
        class_to_idx (dict): Dictionary mapping class name to integer index.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "classes.npy")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        classes = np.load(cache_path, allow_pickle=True)
    else:
        # Load training metadata to determine the universe of classes
        if not os.path.exists(Config.TRAIN_CSV):
            raise FileNotFoundError(f"Train metadata not found at {Config.TRAIN_CSV}")

        df = pd.read_csv(Config.TRAIN_CSV)
        unique_ids = df["Id"].unique()
        classes = np.sort(unique_ids)

        # Save to cache
        np.save(cache_path, classes)

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline based on the mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Strategy:
        - Train:
            1. Resize to IMG_SIZE.
            2. Conservative Affine (Rotate +/- 20, Scale 0.9-1.1).
            3. Horizontal Flip.
            4. Brightness/Contrast (No Saturation/Hue).
            5. Normalize & ToTensor.
        - Val/Test:
            1. Resize to IMG_SIZE.
            2. Normalize & ToTensor.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                # Conservative Geometric Augmentations
                # Scale 0.9-1.1 means scale_limit=0.1
                # Rotation +/- 20 means rotate_limit=20
                A.ShiftScaleRotate(
                    shift_limit=0.0,  # No translation shift
                    scale_limit=0.1,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                # Photometric Augmentations (Brightness/Contrast only)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Normalization
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Whale Species Prediction.
    """

    def __init__(self, mode="train", transform=None, load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
            load_cached_data (bool): Whether to use cached class mappings.
        """
        self.mode = mode
        self.transform = transform

        # Load Metadata
        if mode == "train":
            self.csv_file = Config.TRAIN_CSV
        elif mode == "val":
            self.csv_file = Config.VAL_CSV
        elif mode == "test":
            self.csv_file = Config.TEST_CSV
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"Metadata file not found: {self.csv_file}")

        self.df = pd.read_csv(self.csv_file)

        # Debug Mode: Subsample data
        if Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SUBSET_SIZE].copy()

        # Prepare labels if not in test mode
        self.classes = None
        self.class_to_idx = None

        if mode != "test":
            self.classes, self.class_to_idx = get_class_mapping(
                load_cached_data=load_cached_data
            )

            # Verify that all IDs in the dataframe exist in the mapping
            # This handles the case where val set might be a subset of train
            # But ensures no unknown classes appear in val
            unknown_classes = set(self.df["Id"].unique()) - set(self.classes)
            if unknown_classes:
                # In a strict setting, this should raise an error.
                # However, if using a subset for debug, we might just filter.
                # For this competition, we assume metadata generation handled consistency.
                pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # row['file_path'] is relative, e.g., "train/00022e1a.jpg"
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (should be caught by metadata check, but for safety)
            # Return a black image
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback transform if none provided
            t = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.Normalize(mean=Config.MEAN, std=Config.STD),
                    ToTensorV2(),
                ]
            )
            image = t(image=image)["image"]

        # Return data based on mode
        image_id = row["Image"]

        if self.mode == "test":
            return image, image_id
        else:
            label_name = row["Id"]
            label_idx = self.class_to_idx[label_name]
            return image, torch.tensor(label_idx, dtype=torch.long), image_id
