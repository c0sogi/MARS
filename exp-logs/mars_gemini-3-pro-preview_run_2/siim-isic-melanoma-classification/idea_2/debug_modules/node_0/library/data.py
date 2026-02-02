import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from library.config import Config


def get_transforms(data_split):
    """
    Returns the Albumentations transform pipeline for a given data split.
    """
    if data_split == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


def process_and_cache_metadata(load_cached_data=True):
    """
    Processes metadata (tabular features) and caches the result to disk.
    If cache exists and load_cached_data is True, loads from disk.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    cache_files = {
        "train": os.path.join(Config.WORK_DIR, "meta_train.npy"),
        "val": os.path.join(Config.WORK_DIR, "meta_val.npy"),
        "test": os.path.join(Config.WORK_DIR, "meta_test.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and all_cached:
        print("Loading cached metadata features...")
        meta_train = np.load(cache_files["train"])
        meta_val = np.load(cache_files["val"])
        meta_test = np.load(cache_files["test"])
        return meta_train, meta_val, meta_test

    print("Processing metadata features from scratch...")

    # Load raw CSVs
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # Define features
    num_features = ["age_approx"]
    cat_features = ["sex", "anatom_site_general_challenge"]

    # Define preprocessing pipelines
    # Numerical: Impute missing with mean -> Standardize
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical: Impute missing with 'unknown' -> OneHot
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )

    # Fit on TRAIN only to prevent leakage
    print("Fitting metadata preprocessor on training set...")
    preprocessor.fit(df_train)

    # Transform all sets
    meta_train = preprocessor.transform(df_train).astype(np.float32)
    meta_val = preprocessor.transform(df_val).astype(np.float32)
    meta_test = preprocessor.transform(df_test).astype(np.float32)

    # Save to cache
    print(f"Saving processed metadata to {Config.WORK_DIR}...")
    np.save(cache_files["train"], meta_train)
    np.save(cache_files["val"], meta_val)
    np.save(cache_files["test"], meta_test)

    return meta_train, meta_val, meta_test


class MelanomaDataset(Dataset):
    def __init__(
        self,
        df,
        meta_features,
        transforms=None,
        is_test=False,
        input_root=Config.INPUT_ROOT,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing file paths and targets.
            meta_features (np.ndarray): Pre-processed tabular features aligned with df.
            transforms (albumentations.Compose): Image transformations.
            is_test (bool): If True, returns dummy target.
            input_root (str): Root directory for images.
        """
        self.df = df
        self.meta_features = meta_features
        self.transforms = transforms
        self.is_test = is_test
        self.input_root = input_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # file_path in metadata is relative, e.g., "jpeg/train/ISIC_xxxx.jpg"
        full_path = os.path.join(self.input_root, row["file_path"])

        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (should be rare based on EDA)
            # Create a black image of expected size
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # 3. Get Metadata
        meta = torch.tensor(self.meta_features[idx], dtype=torch.float32)

        # 4. Get Target
        if self.is_test:
            target = torch.tensor(0.0, dtype=torch.float32)
        else:
            target = torch.tensor(row["target"], dtype=torch.float32)

        return image, meta, target


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached metadata features.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load Metadata DataFrames
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # 2. Process/Load Tabular Features
    meta_train, meta_val, meta_test = process_and_cache_metadata(load_cached_data)

    # 3. Handle Debug Mode
    if debug:
        print(f"Debug mode enabled: using {Config.DEBUG_SAMPLE_SIZE} samples.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        meta_train = meta_train[: Config.DEBUG_SAMPLE_SIZE]

        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        meta_val = meta_val[: Config.DEBUG_SAMPLE_SIZE]

        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)
        meta_test = meta_test[: Config.DEBUG_SAMPLE_SIZE]

    # 4. Create Datasets
    train_dataset = MelanomaDataset(
        df=df_train,
        meta_features=meta_train,
        transforms=get_transforms("train"),
        is_test=False,
    )

    val_dataset = MelanomaDataset(
        df=df_val,
        meta_features=meta_val,
        transforms=get_transforms("val"),
        is_test=False,
    )

    test_dataset = MelanomaDataset(
        df=df_test,
        meta_features=meta_test,
        transforms=get_transforms("test"),
        is_test=True,
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
