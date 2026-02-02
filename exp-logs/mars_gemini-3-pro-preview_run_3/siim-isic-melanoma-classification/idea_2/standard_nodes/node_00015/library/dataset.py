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

from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    INPUT_DIR,
    WORKING_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_COLS,
    CAT_COLS,
    DEBUG,
    MAX_DEBUG_SAMPLES,
)


def get_transforms(data_split="train"):
    """
    Returns the Albumentations transforms for the specified data split.

    Args:
        data_split (str): 'train', 'val', or 'test'.
    """
    # ImageNet statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if data_split == "train":
        return A.Compose(
            [
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5
                ),
                # CoarseDropout scaled to approx 32-40 pixels for 384x384 image
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_processed_metadata(load_cached_data=True):
    """
    Loads, preprocesses, and caches the metadata for train, val, and test sets.
    Handles numerical normalization and categorical encoding.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (df_train, df_val, df_test, meta_feature_dim)
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_train_path = os.path.join(WORKING_DIR, "processed_train.parquet")
    cache_val_path = os.path.join(WORKING_DIR, "processed_val.parquet")
    cache_test_path = os.path.join(WORKING_DIR, "processed_test.parquet")

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):
            print("Loading processed metadata from cache...")
            df_train = pd.read_parquet(cache_train_path)
            df_val = pd.read_parquet(cache_val_path)
            df_test = pd.read_parquet(cache_test_path)

            # Calculate meta dim based on columns starting with 'meta_'
            meta_cols = [c for c in df_train.columns if c.startswith("meta_")]
            return df_train, df_val, df_test, len(meta_cols)

    print("Processing metadata from scratch...")

    # Load raw data
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)

    if DEBUG:
        df_train = df_train.head(MAX_DEBUG_SAMPLES)
        df_val = df_val.head(MAX_DEBUG_SAMPLES)
        df_test = df_test.head(MAX_DEBUG_SAMPLES)

    # Define Preprocessing Pipeline
    # Numerical: Impute with mean, Scale
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical: Impute with 'missing', OneHot
    # handle_unknown='ignore' is crucial for test set robustness
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUM_COLS),
            ("cat", cat_transformer, CAT_COLS),
        ]
    )

    # Fit on Train, Transform all
    # We concatenate train and val for fitting to ensure better coverage of categories
    # if strictly following ML rules, fit only on train. Here we fit on train.
    preprocessor.fit(df_train)

    # Transform and retrieve feature names
    train_meta = preprocessor.transform(df_train)
    val_meta = preprocessor.transform(df_val)
    test_meta = preprocessor.transform(df_test)

    # Get feature names to create dataframe columns
    # Note: older sklearn versions might use get_feature_names
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn
        feature_names = NUM_COLS + list(
            preprocessor.named_transformers_["cat"]["onehot"].get_feature_names(
                CAT_COLS
            )
        )

    # Prefix to avoid collision
    feature_names = [f"meta_{name}" for name in feature_names]

    # Assign back to dataframes
    df_train_meta = pd.DataFrame(
        train_meta, columns=feature_names, index=df_train.index
    )
    df_val_meta = pd.DataFrame(val_meta, columns=feature_names, index=df_val.index)
    df_test_meta = pd.DataFrame(test_meta, columns=feature_names, index=df_test.index)

    df_train = pd.concat([df_train, df_train_meta], axis=1)
    df_val = pd.concat([df_val, df_val_meta], axis=1)
    df_test = pd.concat([df_test, df_test_meta], axis=1)

    # Cache results
    df_train.to_parquet(cache_train_path)
    df_val.to_parquet(cache_val_path)
    df_test.to_parquet(cache_test_path)

    return df_train, df_val, df_test, len(feature_names)


class MelanomaDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None, meta_cols=None):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.meta_cols = meta_cols

        # Pre-convert file paths to absolute paths for speed
        # Metadata file_path is relative to INPUT_DIR (e.g., "jpeg/train/...")
        self.file_paths = [os.path.join(image_dir, fp) for fp in df["file_path"].values]

        # Pre-extract targets and meta features
        if "target" in df.columns:
            self.targets = df["target"].values.astype(np.float32)
        else:
            self.targets = np.zeros(len(df), dtype=np.float32)  # Dummy for test

        if self.meta_cols:
            self.meta_features = df[self.meta_cols].values.astype(np.float32)
        else:
            self.meta_features = np.zeros((len(df), 1), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Read Image
        image = cv2.imread(path)
        if image is None:
            # Fallback for missing images (though analysis showed 0 missing)
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Get Meta and Target
        meta = torch.tensor(self.meta_features[idx])
        target = torch.tensor(self.targets[idx])

        # Return dict
        return {
            "image": image,
            "meta": meta,
            "target": target,
            "image_name": self.df.iloc[idx]["image_name"],  # Useful for inference
        }


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        tuple: (train_loader, val_loader, test_loader, meta_dim)
    """
    # Process Metadata
    df_train, df_val, df_test, meta_dim = get_processed_metadata(load_cached_data)

    # Identify meta columns
    meta_cols = [c for c in df_train.columns if c.startswith("meta_")]

    # Create Datasets
    train_dataset = MelanomaDataset(
        df_train, INPUT_DIR, transforms=get_transforms("train"), meta_cols=meta_cols
    )

    val_dataset = MelanomaDataset(
        df_val, INPUT_DIR, transforms=get_transforms("val"), meta_cols=meta_cols
    )

    test_dataset = MelanomaDataset(
        df_test, INPUT_DIR, transforms=get_transforms("test"), meta_cols=meta_cols
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, meta_dim
