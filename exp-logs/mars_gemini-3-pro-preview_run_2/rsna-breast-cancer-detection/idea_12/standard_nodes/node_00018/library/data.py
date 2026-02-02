import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from library import config
from library import utils


class BreastCancerDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-extract paths and labels for speed
        self.file_paths = df["file_path"].values
        # Tabular data preparation
        # We assume df has already been encoded/normalized by process_metadata
        self.cat_features = df[config.CATEGORICAL_COLS].values.astype(np.int64)
        self.num_features = df[config.NUMERICAL_COLS].values.astype(np.float32)

        # Targets
        if "cancer" in df.columns:
            self.labels = df["cancer"].values.astype(np.float32)
        else:
            self.labels = np.zeros(len(df), dtype=np.float32)  # Dummy for test

        # Prediction IDs for submission grouping (only needed for test really, but good to have)
        if "prediction_id" in df.columns:
            self.prediction_ids = df["prediction_id"].values
        else:
            # Fallback for train/val
            self.prediction_ids = [
                f"{pid}_{lat}" for pid, lat in zip(df["patient_id"], df["laterality"])
            ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Image Processing
        rel_path = self.file_paths[idx]
        full_path = os.path.join(config.INPUT_DIR, rel_path)

        # Use the robust byte-level reader from utils
        img = utils.read_image_from_bytes(full_path)

        # Handle missing or corrupt images
        if img is None:
            # Create a black image of default size if load fails
            img = np.zeros((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]), dtype=np.uint8)

        # Ensure image is single channel (grayscale)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize to 8-bit if 16-bit (common in DICOM)
        if img.dtype != np.uint8:
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min) * 255.0
            else:
                img = img - img_min
            img = img.astype(np.uint8)

        # --- Simulated Windowing Strategy ---
        # Channel 1: Linear / Standard Grayscale
        c1 = img

        # Channel 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        c2 = clahe.apply(img)

        # Channel 3: Gamma Correction (Gamma=0.5 to expand dark/dense areas)
        gamma = 0.5
        invGamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        c3 = cv2.LUT(img, table)

        # Stack to 3 channels (H, W, 3)
        image_stacked = np.dstack([c1, c2, c3])

        # Apply Augmentations/Transforms
        if self.transforms:
            augmented = self.transforms(image=image_stacked)
            image_tensor = augmented["image"]
        else:
            # Fallback
            image_tensor = (
                torch.from_numpy(image_stacked.transpose(2, 0, 1)).float() / 255.0
            )

        # 2. Tabular Processing
        cat_feats = torch.tensor(self.cat_features[idx], dtype=torch.long)
        num_feats = torch.tensor(self.num_features[idx], dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Return: image, (cat, num), label
        return image_tensor, (cat_feats, num_feats), label


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for train or validation/test.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), p=0.5),
                # Normalize using ImageNet statistics as we use pretrained backbone
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def process_metadata(load_cached_data=True):
    """
    Loads metadata, performs encoding/normalization, and caches the result.
    Strictly follows the caching logic requirement using Parquet.
    """
    cache_path_train = os.path.join(config.CACHE_DIR, "processed_train.parquet")
    cache_path_val = os.path.join(config.CACHE_DIR, "processed_val.parquet")
    cache_path_test = os.path.join(config.CACHE_DIR, "processed_test.parquet")
    cache_path_meta = os.path.join(config.CACHE_DIR, "feature_meta.npy")

    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(cache_path_train)
        and os.path.exists(cache_path_val)
        and os.path.exists(cache_path_test)
        and os.path.exists(cache_path_meta)
    ):

        print("Loading cached processed metadata...")
        train_df = pd.read_parquet(cache_path_train)
        val_df = pd.read_parquet(cache_path_val)
        test_df = pd.read_parquet(cache_path_test)
        feature_meta = np.load(cache_path_meta, allow_pickle=True).item()
        return train_df, val_df, test_df, feature_meta

    print("Processing metadata from scratch...")

    # Load raw CSVs
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # --- Tabular Preprocessing ---

    # 1. Handle Missing Values
    # Numerical: Fill with median (robust to outliers)
    for col in config.NUMERICAL_COLS:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        val_df[col] = val_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

    # Categorical: Fill with 'unknown'
    for col in config.CATEGORICAL_COLS:
        train_df[col] = train_df[col].fillna("unknown").astype(str)
        val_df[col] = val_df[col].fillna("unknown").astype(str)
        test_df[col] = test_df[col].fillna("unknown").astype(str)

    # 2. Encoding & Normalization
    vocab_sizes = {}

    # Categorical: Ordinal Encoding
    # We fit on TRAIN and transform others. Handle unknown by assigning a new index.
    for col in config.CATEGORICAL_COLS:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        # Fit on train
        train_vals = train_df[[col]].values
        encoder.fit(train_vals)

        # Transform
        train_df[col] = encoder.transform(train_vals).astype(int)
        val_df[col] = encoder.transform(val_df[[col]].values).astype(int)
        test_df[col] = encoder.transform(test_df[[col]].values).astype(int)

        # Handle unknowns (mapped to -1): Map them to the last index (vocab_size)
        n_classes = len(encoder.categories_[0])

        # Shift -1 to n_classes
        train_df[col] = train_df[col].apply(lambda x: n_classes if x == -1 else x)
        val_df[col] = val_df[col].apply(lambda x: n_classes if x == -1 else x)
        test_df[col] = test_df[col].apply(lambda x: n_classes if x == -1 else x)

        # Vocab size is n_classes + 1 (for the unknown token)
        vocab_sizes[col] = n_classes + 1

    # Numerical: Standard Scaling
    scaler = StandardScaler()
    scaler.fit(train_df[config.NUMERICAL_COLS])

    train_df[config.NUMERICAL_COLS] = scaler.transform(train_df[config.NUMERICAL_COLS])
    val_df[config.NUMERICAL_COLS] = scaler.transform(val_df[config.NUMERICAL_COLS])
    test_df[config.NUMERICAL_COLS] = scaler.transform(test_df[config.NUMERICAL_COLS])

    # Save to Cache
    print("Saving processed metadata to cache...")
    train_df.to_parquet(cache_path_train)
    val_df.to_parquet(cache_path_val)
    test_df.to_parquet(cache_path_test)

    feature_meta = {"vocab_sizes": vocab_sizes}
    np.save(cache_path_meta, feature_meta)

    return train_df, val_df, test_df, feature_meta


def get_dataloaders(load_cached_data=True, debug=config.DEBUG):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached metadata.
        debug (bool): If True, subsamples the data for quick debugging.

    Returns:
        train_loader, val_loader, test_loader, feature_meta
    """
    # Load processed data
    train_df, val_df, test_df, feature_meta = process_metadata(
        load_cached_data=load_cached_data
    )

    if debug:
        print(f"DEBUG MODE: Subsampling {config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(config.DEBUG_SAMPLE_SIZE)

    # Create Datasets
    train_dataset = BreastCancerDataset(
        train_df, transforms=get_transforms(mode="train"), mode="train"
    )

    val_dataset = BreastCancerDataset(
        val_df, transforms=get_transforms(mode="val"), mode="val"
    )

    test_dataset = BreastCancerDataset(
        test_df, transforms=get_transforms(mode="test"), mode="test"
    )

    # Create DataLoaders
    # Note: Using standard RandomSampler (shuffle=True) to preserve class priors
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, feature_meta
