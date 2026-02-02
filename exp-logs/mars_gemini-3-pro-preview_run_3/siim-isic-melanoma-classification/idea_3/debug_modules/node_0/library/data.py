import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from library.config import Config
from library.utils import seed_everything


def get_transforms(mode="train", image_size=Config.IMAGE_SIZE):
    """
    Returns the Albumentations transforms based on the mode.

    Args:
        mode (str): 'train' or 'valid'/'test'.
        image_size (int): Target image size.

    Returns:
        A.Compose: Composed transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                # Geometric Augmentations only (as per strategy)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=180, p=0.5),
                # Micro-Hole CoarseDropout (Noise injection)
                A.CoarseDropout(
                    max_holes=12,
                    max_height=int(image_size * 0.05),
                    max_width=int(image_size * 0.05),
                    min_holes=4,
                    min_height=4,
                    min_width=4,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
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


class ISICDataset(Dataset):
    """
    PyTorch Dataset for Skin Lesion Classification.
    Handles image loading and tabular feature extraction.
    """

    def __init__(self, df, transforms=None, mode="train", tabular_cols=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata and file paths.
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train' or 'test'.
            tabular_cols (list): List of column names to use as tabular features.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode
        self.tabular_cols = tabular_cols or []

        # Pre-convert file paths to absolute paths to save time in __getitem__
        # Assuming file_path in df is relative to input dir (e.g., "jpeg/train/...")
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in self.df["file_path"].values
        ]

        # Pre-extract tabular features as float32 array
        if self.tabular_cols:
            self.tabular_data = self.df[self.tabular_cols].values.astype(np.float32)
        else:
            self.tabular_data = np.zeros((len(self.df), 1), dtype=np.float32)

        # Pre-extract targets if training
        if self.mode == "train":
            self.targets = self.df["target"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # 1. Load Image
        path = self.file_paths[index]
        image = cv2.imread(path)

        if image is None:
            # Fallback for missing images (should not happen based on checks)
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # 3. Get Tabular Data
        tabular = torch.tensor(self.tabular_data[index], dtype=torch.float32)

        # 4. Return
        if self.mode == "train":
            target = torch.tensor(self.targets[index], dtype=torch.float32)
            return {
                "image": image,
                "tabular": tabular,
                "target": target,
                "image_name": self.df.iloc[index]["image_name"],
            }
        else:
            return {
                "image": image,
                "tabular": tabular,
                "image_name": self.df.iloc[index]["image_name"],
            }


def process_data(load_cached_data=True):
    """
    Loads metadata, performs tabular feature engineering, creates folds,
    and caches the processed dataframes to disk.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (train_df, test_df, feature_cols)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "processed_train.parquet")
    test_cache_path = os.path.join(cache_dir, "processed_test.parquet")
    meta_cache_path = os.path.join(cache_dir, "feature_cols.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(test_cache_path)
            and os.path.exists(meta_cache_path)
        ):
            try:
                train_df = pd.read_parquet(train_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                feature_cols = np.load(meta_cache_path, allow_pickle=True).tolist()
                return train_df, test_df, feature_cols
            except Exception:
                pass  # Fallback to processing

    # 2. Process from scratch
    seed_everything(Config.SEED)

    # Load raw metadata
    # Strategy: Combine original train and val metadata to use 100% data for K-Fold
    df_train_part = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_part = pd.read_csv(Config.VAL_METADATA_PATH)
    df_train = pd.concat([df_train_part, df_val_part], ignore_index=True)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Tabular Feature Engineering Pipeline
    # Define transformers
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, Config.TABULAR_NUM_COLS),
            ("cat", categorical_transformer, Config.TABULAR_CAT_COLS),
        ],
        verbose_feature_names_out=False,
    )

    # Fit on training data
    preprocessor.fit(df_train)

    # Transform both train and test
    # We need to get feature names to create a dataframe with named columns
    # so we can save/load cleanly.

    # Transform Train
    X_train_processed = preprocessor.transform(df_train)
    feature_names = preprocessor.get_feature_names_out()

    # Create temporary DF for features
    df_train_features = pd.DataFrame(
        X_train_processed, columns=feature_names, index=df_train.index
    )
    # Concatenate back to metadata (excluding original raw cols to avoid dupes if names clash, though they shouldn't)
    df_train_final = pd.concat([df_train, df_train_features], axis=1)

    # Transform Test
    X_test_processed = preprocessor.transform(df_test)
    df_test_features = pd.DataFrame(
        X_test_processed, columns=feature_names, index=df_test.index
    )
    df_test_final = pd.concat([df_test, df_test_features], axis=1)

    # 3. Create Stratified Group K-Fold Splits
    # We need to handle missing patient_ids for grouping
    df_train_final["patient_id"] = df_train_final["patient_id"].fillna(
        "unknown_patient"
    )

    sgkf = StratifiedGroupKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    df_train_final["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(
            df_train_final, df_train_final["target"], df_train_final["patient_id"]
        )
    ):
        df_train_final.loc[val_idx, "fold"] = fold

    # 4. Save to Cache
    df_train_final.to_parquet(train_cache_path, index=False)
    df_test_final.to_parquet(test_cache_path, index=False)
    np.save(meta_cache_path, np.array(feature_names))

    return df_train_final, df_test_final, feature_names.tolist()


def get_loaders(fold, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold.

    Args:
        fold (int): The validation fold index (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        tuple: (train_loader, val_loader, feature_cols_count)
    """
    # Get processed data
    df_train_full, _, feature_cols = process_data(load_cached_data=load_cached_data)

    # Split into Train and Validation based on fold column
    df_train = df_train_full[df_train_full["fold"] != fold].reset_index(drop=True)
    df_val = df_train_full[df_train_full["fold"] == fold].reset_index(drop=True)

    # Create Datasets
    train_dataset = ISICDataset(
        df_train,
        transforms=get_transforms(mode="train"),
        mode="train",
        tabular_cols=feature_cols,
    )

    val_dataset = ISICDataset(
        df_val,
        transforms=get_transforms(mode="valid"),
        mode="train",  # 'train' mode returns targets
        tabular_cols=feature_cols,
    )

    # Create Loaders
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
        drop_last=False,
    )

    return train_loader, val_loader, len(feature_cols)


def get_test_loader(load_cached_data=True):
    """
    Creates DataLoader for the test set.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        tuple: (test_loader, feature_cols_count)
    """
    _, df_test, feature_cols = process_data(load_cached_data=load_cached_data)

    test_dataset = ISICDataset(
        df_test,
        transforms=get_transforms(mode="test"),
        mode="test",
        tabular_cols=feature_cols,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return test_loader, len(feature_cols)
