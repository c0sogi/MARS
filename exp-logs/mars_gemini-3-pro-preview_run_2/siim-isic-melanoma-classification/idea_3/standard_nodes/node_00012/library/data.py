import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from library.config import Config


def get_transforms(data="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): 'train' for augmentation, 'val' or 'test' for resizing/normalization only.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5
                ),
                A.Normalize(
                    mean=Config.IMG_MEAN,
                    std=Config.IMG_STD,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=Config.IMG_MEAN,
                    std=Config.IMG_STD,
                ),
                ToTensorV2(),
            ]
        )


def process_metadata(df_train, df_val, df_test, load_cached_data=True):
    """
    Processes metadata: Imputation, Scaling, and One-Hot Encoding.
    Handles caching of processed arrays to disk.

    Args:
        df_train, df_val, df_test: DataFrames for the respective splits.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (meta_train, meta_val, meta_test) as numpy arrays.
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    path_train = os.path.join(cache_dir, "meta_train.npy")
    path_val = os.path.join(cache_dir, "meta_val.npy")
    path_test = os.path.join(cache_dir, "meta_test.npy")

    # Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(path_train)
            and os.path.exists(path_val)
            and os.path.exists(path_test)
        ):
            try:
                meta_train = np.load(path_train)
                meta_val = np.load(path_val)
                meta_test = np.load(path_test)
                return meta_train, meta_val, meta_test
            except Exception:
                # If load fails, proceed to recompute
                pass

    # Define Transformers
    numerical_cols = Config.NUMERICAL_COLS
    categorical_cols = Config.CATEGORICAL_COLS

    # Numerical: Impute Mean -> Standardize
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical: Impute Mode -> OneHot
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Fit on Train ONLY
    preprocessor.fit(df_train[numerical_cols + categorical_cols])

    # Transform all sets
    meta_train = preprocessor.transform(
        df_train[numerical_cols + categorical_cols]
    ).astype(np.float32)
    meta_val = preprocessor.transform(df_val[numerical_cols + categorical_cols]).astype(
        np.float32
    )
    meta_test = preprocessor.transform(
        df_test[numerical_cols + categorical_cols]
    ).astype(np.float32)

    # Save to cache
    np.save(path_train, meta_train)
    np.save(path_val, meta_val)
    np.save(path_test, meta_test)

    return meta_train, meta_val, meta_test


class ISICDataset(Dataset):
    def __init__(self, df, meta_data, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing file paths and targets.
            meta_data (np.ndarray): Processed metadata features corresponding to df rows.
            transform (albumentations.Compose): Augmentation pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.meta_data = meta_data
        self.transform = transform
        self.mode = mode

        # Pre-compute full paths to avoid overhead in __getitem__
        # file_path in df is relative to input root (e.g., "jpeg/train/ISIC_xxxx.jpg")
        self.file_paths = [
            os.path.join(Config.INPUT_ROOT, p) for p in df["file_path"].values
        ]

        # Targets
        if self.mode != "test":
            self.targets = df["target"].values
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load Image
        image_path = self.file_paths[idx]
        image = cv2.imread(image_path)

        if image is None:
            # Fallback for missing images (should be handled by EDA, but for safety)
            # Create a black image
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Get Metadata
        meta = torch.tensor(self.meta_data[idx], dtype=torch.float32)

        # Return format: ((image, metadata), target)
        if self.mode != "test":
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return (image, meta), target
        else:
            # For test, return dummy target or just inputs.
            # To keep signature consistent for generic loops, we can return a dummy target.
            return (image, meta), torch.tensor(0.0, dtype=torch.float32)


def get_loaders(load_cached_data=True, debug=False):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached metadata processing.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata DataFrames
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    if debug or Config.DEBUG:
        df_train = df_train.head(100)
        df_val = df_val.head(50)
        df_test = df_test.head(50)

    # Process Metadata
    meta_train, meta_val, meta_test = process_metadata(
        df_train, df_val, df_test, load_cached_data=load_cached_data
    )

    # Create Datasets
    train_dataset = ISICDataset(
        df_train, meta_train, transform=get_transforms("train"), mode="train"
    )

    val_dataset = ISICDataset(
        df_val, meta_val, transform=get_transforms("val"), mode="val"
    )

    test_dataset = ISICDataset(
        df_test, meta_test, transform=get_transforms("test"), mode="test"
    )

    # Create DataLoaders
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
