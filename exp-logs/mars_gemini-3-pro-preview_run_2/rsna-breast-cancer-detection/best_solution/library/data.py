import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from library.config import Config
from library.utils import read_dicom_bytes


def get_transforms(phase: str):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    height, width = Config.IMG_SIZE

    if phase == "train":
        return A.Compose(
            [
                A.Resize(height, width),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                # CoarseDropout acts as regularization, similar to Cutout
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height, width),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


class BreastCancerDataset(Dataset):
    """
    PyTorch Dataset for Breast Cancer Detection.
    Handles loading of DICOM images (via byte stream) and pre-processed tabular features.
    """

    def __init__(self, df, tabular_data, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            tabular_data (np.ndarray): Pre-processed tabular features matching df rows.
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.tabular_data = tabular_data
        self.transforms = transforms
        self.mode = mode

        # Pre-compute full paths
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].values
        ]

        # Prepare targets if not in test mode
        if self.mode != "test":
            self.targets = df["cancer"].values.astype(np.float32)

        # Prepare IDs for tracking/submission
        if "prediction_id" in df.columns:
            self.ids = df["prediction_id"].values
        else:
            # Fallback for train/val where prediction_id might not exist
            self.ids = df["patient_id"].astype(str).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        path = self.file_paths[idx]
        img = read_dicom_bytes(path)

        # 2. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]

        # 3. Get Tabular Features
        tab_features = torch.tensor(self.tabular_data[idx], dtype=torch.float32)

        # 4. Return based on mode
        if self.mode == "test":
            sample_id = self.ids[idx]
            return img, tab_features, sample_id
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return img, tab_features, target


def process_tabular_data(train_df, val_df, test_df):
    """
    Fits preprocessors on training data and transforms train, val, and test sets.
    Uses OneHotEncoder for categoricals and StandardScaler for numericals.

    Returns:
        tuple: (train_processed, val_processed, test_processed, preprocessor)
    """
    num_features = Config.NUMERICAL_FEATURES
    cat_features = Config.CATEGORICAL_FEATURES

    # Ensure categorical columns are strings
    for df in [train_df, val_df, test_df]:
        for col in cat_features:
            df[col] = df[col].astype(str)

    # Numerical Pipeline: Impute missing -> Scale
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical Pipeline: Impute missing -> OneHot
    # handle_unknown='ignore' ensures robustness against unseen labels in test
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )

    # Fit only on training data
    preprocessor.fit(train_df)

    # Transform all sets
    train_processed = preprocessor.transform(train_df)
    val_processed = preprocessor.transform(val_df)
    test_processed = preprocessor.transform(test_df)

    return train_processed, val_processed, test_processed, preprocessor


def get_dataloaders(debug=False):
    """
    Factory function to create DataLoaders for Train, Val, and Test.

    Args:
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # Process Tabular Data
    train_tab, val_tab, test_tab, _ = process_tabular_data(train_df, val_df, test_df)

    # Instantiate Datasets
    train_dataset = BreastCancerDataset(
        train_df, train_tab, transforms=get_transforms("train"), mode="train"
    )

    val_dataset = BreastCancerDataset(
        val_df, val_tab, transforms=get_transforms("val"), mode="val"
    )

    test_dataset = BreastCancerDataset(
        test_df, test_tab, transforms=get_transforms("test"), mode="test"
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
