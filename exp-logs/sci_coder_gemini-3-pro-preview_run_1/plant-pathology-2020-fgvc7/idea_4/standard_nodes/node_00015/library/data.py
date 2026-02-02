import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import CFG


def get_transforms(data: str):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): 'train' for augmentation, 'valid' or 'test' for resizing/normalization only.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(CFG.img_size, CFG.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(CFG.img_size, CFG.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, file_path, targets).
            transform (albumentations.Compose): Transformations to apply to the image.
        """
        self.df = df
        self.transform = transform

        # Determine if labels are present in the dataframe
        self.target_cols = CFG.target_cols
        self.has_labels = all(col in df.columns for col in self.target_cols)

        # Pre-fetch paths and labels (if available) to avoid overhead in __getitem__
        self.file_paths = df["file_path"].values
        if self.has_labels:
            self.labels = df[self.target_cols].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(CFG.input_root, rel_path)

        # Load image using OpenCV
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (though metadata check should prevent this)
            # Create a blank image
            image = np.zeros((CFG.img_size, CFG.img_size, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return image and label
        if self.has_labels:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            # If no labels (test set), return image and a dummy tensor
            # The caller can rely on the dataframe order or index
            return image, torch.zeros(len(self.target_cols), dtype=torch.float32)


def load_full_train_data(debug: bool = CFG.debug) -> pd.DataFrame:
    """
    Loads and merges training and validation metadata for K-Fold Cross Validation.

    Args:
        debug (bool): If True, returns a small subset of the data.

    Returns:
        pd.DataFrame: Merged DataFrame containing all training data.
    """
    if not os.path.exists(CFG.train_metadata_path) or not os.path.exists(
        CFG.val_metadata_path
    ):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation script has run."
        )

    df_train = pd.read_csv(CFG.train_metadata_path)
    df_val = pd.read_csv(CFG.val_metadata_path)

    # Concatenate to form the full dataset for CV
    df_full = pd.concat([df_train, df_val], ignore_index=True)

    if debug:
        print("Debug mode enabled: Subsampling data.")
        df_full = df_full.sample(
            n=min(len(df_full), 50), random_state=CFG.seed
        ).reset_index(drop=True)

    return df_full


def load_test_data(debug: bool = CFG.debug) -> pd.DataFrame:
    """
    Loads test metadata.

    Args:
        debug (bool): If True, returns a small subset of the data.

    Returns:
        pd.DataFrame: DataFrame containing test data.
    """
    if not os.path.exists(CFG.test_metadata_path):
        raise FileNotFoundError(
            f"Test metadata file not found at {CFG.test_metadata_path}"
        )

    df_test = pd.read_csv(CFG.test_metadata_path)

    if debug:
        df_test = df_test.sample(
            n=min(len(df_test), 20), random_state=CFG.seed
        ).reset_index(drop=True)

    return df_test
