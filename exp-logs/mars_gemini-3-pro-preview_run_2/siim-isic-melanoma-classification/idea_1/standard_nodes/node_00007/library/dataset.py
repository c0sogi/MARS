import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from library.config import Config
from library.utils import seed_everything


class MelanomaDataset(Dataset):
    """
    PyTorch Dataset for Melanoma Detection.
    Handles loading of images, basic preprocessing (resize/normalize),
    and integration of pre-processed metadata features.
    """

    def __init__(self, df, meta_features, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing image paths and targets.
            meta_features (np.ndarray): Pre-processed metadata features aligned with df.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.meta_features = meta_features.astype(np.float32)
        self.mode = mode

        # Define Augmentation Pipeline (Cite solution_lesson_node_00006)
        if self.mode == "train":
            self.transform = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=20, p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        # Pre-compute full paths to avoid overhead in __getitem__
        # Config.INPUT_ROOT is "./input", file_path in df is like "jpeg/train/..."
        self.file_paths = [
            os.path.join(Config.INPUT_ROOT, fp) for fp in df["file_path"].values
        ]

        if self.mode != "test":
            self.targets = df["target"].values.astype(np.float32)
        else:
            self.targets = np.zeros(len(df), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        path = self.file_paths[idx]
        img = cv2.imread(path)

        if img is None:
            # Fallback for missing/corrupt images: create black image
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Apply Augmentations
        augmented = self.transform(image=img)
        img = augmented["image"]

        # 3. Transpose to CHW (PyTorch format)
        img = img.transpose(2, 0, 1)

        # 4. Convert to Tensors
        img_tensor = torch.from_numpy(img)
        meta_tensor = torch.from_numpy(self.meta_features[idx])
        target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)

        return img_tensor, meta_tensor, target_tensor


def preprocess_metadata(load_cached_data=True):
    """
    Loads metadata CSVs, performs feature engineering (imputation, scaling, OHE),
    and caches the resulting numpy arrays.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        tuple: (df_train, meta_train, df_val, meta_val, df_test, meta_test)
    """
    # Cache file paths
    cache_train = os.path.join(Config.WORKING_DIR, "meta_train.npy")
    cache_val = os.path.join(Config.WORKING_DIR, "meta_val.npy")
    cache_test = os.path.join(Config.WORKING_DIR, "meta_test.npy")

    # Load raw dataframes
    df_train = pd.read_csv(Config.TRAIN_CSV_PATH)
    df_val = pd.read_csv(Config.VAL_CSV_PATH)
    df_test = pd.read_csv(Config.TEST_CSV_PATH)

    # Check cache
    if (
        load_cached_data
        and os.path.exists(cache_train)
        and os.path.exists(cache_val)
        and os.path.exists(cache_test)
    ):

        meta_train = np.load(cache_train)
        meta_val = np.load(cache_val)
        meta_test = np.load(cache_test)

        # Verify alignment (simple length check)
        if (
            len(meta_train) == len(df_train)
            and len(meta_val) == len(df_val)
            and len(meta_test) == len(df_test)
        ):
            return df_train, meta_train, df_val, meta_val, df_test, meta_test

    # If cache miss or force recompute:

    # Define Transformers
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

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, Config.META_NUM_COLS),
            ("cat", categorical_transformer, Config.META_CAT_COLS),
        ]
    )

    # Fit on TRAIN only
    preprocessor.fit(df_train)

    # Transform all sets
    meta_train = preprocessor.transform(df_train)
    meta_val = preprocessor.transform(df_val)
    meta_test = preprocessor.transform(df_test)

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_train, meta_train)
    np.save(cache_val, meta_val)
    np.save(cache_test, meta_test)

    return df_train, meta_train, df_val, meta_val, df_test, meta_test


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    Handles metadata preprocessing and debugging sampling.

    Args:
        load_cached_data (bool): Whether to use cached metadata features.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    # 1. Get Data
    df_train, meta_train, df_val, meta_val, df_test, meta_test = preprocess_metadata(
        load_cached_data
    )

    # 2. Handle Debug Mode (Subsampling)
    if Config.DEBUG:
        # Slice the dataframes and arrays
        n_train = min(len(df_train), Config.DEBUG_SAMPLE_SIZE)
        n_val = min(len(df_val), Config.DEBUG_SAMPLE_SIZE)
        n_test = min(len(df_test), Config.DEBUG_SAMPLE_SIZE)

        df_train = df_train.iloc[:n_train].reset_index(drop=True)
        meta_train = meta_train[:n_train]

        df_val = df_val.iloc[:n_val].reset_index(drop=True)
        meta_val = meta_val[:n_val]

        df_test = df_test.iloc[:n_test].reset_index(drop=True)
        meta_test = meta_test[:n_test]

    # 3. Create Datasets
    train_dataset = MelanomaDataset(df_train, meta_train, mode="train")
    val_dataset = MelanomaDataset(df_val, meta_val, mode="val")
    test_dataset = MelanomaDataset(df_test, meta_test, mode="test")

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
