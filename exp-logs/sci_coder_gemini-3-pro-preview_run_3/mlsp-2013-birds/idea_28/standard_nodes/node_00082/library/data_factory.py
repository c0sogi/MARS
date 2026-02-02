import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skmultilearn.model_selection import IterativeStratification

from library.config import Config
from library.utils import seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)


def load_histogram_features(load_cached_data=True):
    """
    Parses the histogram_of_segments.txt file into a DataFrame.
    Caches the result to parquet for faster loading.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "histogram_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Parse the text file
    # Format: rec_id,val1,val2,...
    data = []
    if os.path.exists(Config.HISTOGRAM_FILE):
        with open(Config.HISTOGRAM_FILE, "r") as f:
            lines = f.readlines()
            # Skip header if exists (it says rec_id,[histogram...])
            start_idx = 1 if "rec_id" in lines[0] else 0

            for line in lines[start_idx:]:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                rec_id = int(parts[0])
                features = [float(x) for x in parts[1:]]
                data.append({"rec_id": rec_id, "features": features})

    df = pd.DataFrame(data)

    # Save to cache
    if not df.empty:
        df.to_parquet(cache_path, index=False)

    return df


def create_folds(load_cached_data=True):
    """
    Combines train and val metadata, then performs Iterative Stratified K-Fold.
    Returns a DataFrame with a 'fold' column.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "folds.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Combine to form the full development set
    df = pd.concat([train_df, val_df], ignore_index=True)

    # Prepare X and y for stratification
    # We only need y for stratification, X can be indices
    X = df.index.values.reshape(-1, 1)

    # Parse labels to binary matrix
    num_samples = len(df)
    num_classes = Config.NUM_CLASSES
    y = np.zeros((num_samples, num_classes), dtype=int)

    for idx, row in df.iterrows():
        if pd.notna(row["labels"]) and row["labels"] != "?":
            lbls = [int(x) for x in str(row["labels"]).split()]
            for lbl in lbls:
                if 0 <= lbl < num_classes:
                    y[idx, lbl] = 1

    # Perform Iterative Stratified K-Fold
    k_fold = IterativeStratification(n_splits=Config.N_FOLDS, order=1)

    df["fold"] = -1

    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X, y)):
        df.loc[val_indices, "fold"] = fold_idx

    # Save to cache
    df.to_parquet(cache_path, index=False)

    return df


class BirdDataset(Dataset):
    """
    Unified Dataset for both Image (CNN) and Feature (MLP) streams.
    """

    def __init__(self, df, feature_df=None, mode="train", transform=None):
        self.df = df.reset_index(drop=True)
        self.feature_df = feature_df
        self.mode = mode
        self.transform = transform

        # Pre-process feature lookup
        self.feature_map = {}
        if self.feature_df is not None:
            for _, row in self.feature_df.iterrows():
                self.feature_map[int(row["rec_id"])] = np.array(
                    row["features"], dtype=np.float32
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = int(row["rec_id"])

        # -----------------------------------------------------------
        # 1. Image Processing (Spectrogram)
        # -----------------------------------------------------------
        # Construct path: essential_data/src_wavs/X.wav -> supplemental_data/spectrograms/X.bmp
        wav_rel_path = row["file_path"]
        filename = os.path.basename(wav_rel_path)
        filename_no_ext = os.path.splitext(filename)[0]
        bmp_path = os.path.join(Config.SPECTROGRAM_DIR, f"{filename_no_ext}.bmp")

        # Load Image
        if os.path.exists(bmp_path):
            # Load as BGR (3 channels)
            image = cv2.imread(bmp_path, cv2.IMREAD_COLOR)
            if image is None:
                # Fallback for corrupt files
                image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Placeholder if missing
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Basic resize and normalize if no transform provided (fallback)
            resizer = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = resizer(image=image)["image"]

        # -----------------------------------------------------------
        # 2. Feature Processing (Histogram)
        # -----------------------------------------------------------
        if rec_id in self.feature_map:
            features = self.feature_map[rec_id].copy()
        else:
            features = np.zeros(Config.MLP_INPUT_DIM, dtype=np.float32)

        # Feature Augmentation (Noise) for training
        if self.mode == "train":
            noise = np.random.normal(0, 0.01, features.shape).astype(np.float32)
            features = features + noise

        # -----------------------------------------------------------
        # 3. Label Processing
        # -----------------------------------------------------------
        label_vec = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
        if "labels" in row and pd.notna(row["labels"]) and row["labels"] != "?":
            lbls = [int(x) for x in str(row["labels"]).split()]
            for lbl in lbls:
                if 0 <= lbl < Config.NUM_CLASSES:
                    label_vec[lbl] = 1.0

        return {
            "image": image,
            "features": torch.tensor(features, dtype=torch.float32),
            "target": torch.tensor(label_vec, dtype=torch.float32),
            "rec_id": torch.tensor(rec_id, dtype=torch.long),
        }


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                # Safe-Zone Horizontal Translation (<10%)
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5
                ),
                # Photometric Augmentation
                A.RandomBrightnessContrast(p=0.5),
                # Normalize (ImageNet stats)
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_loaders(
    fold_idx,
    folds_df,
    feature_df,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Creates DataLoaders for a specific fold.

    Args:
        fold_idx (int): The fold index to use for validation.
        folds_df (pd.DataFrame): DataFrame containing 'fold' column.
        feature_df (pd.DataFrame): DataFrame containing histogram features.
    """
    train_df = folds_df[folds_df["fold"] != fold_idx].copy()
    val_df = folds_df[folds_df["fold"] == fold_idx].copy()

    # Debugging: Reduce size
    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    train_dataset = BirdDataset(
        train_df, feature_df=feature_df, mode="train", transform=get_transforms("train")
    )

    val_dataset = BirdDataset(
        val_df, feature_df=feature_df, mode="val", transform=get_transforms("val")
    )

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

    return train_loader, val_loader


def get_test_loader(
    feature_df, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates DataLoader for the test set.
    """
    test_df = pd.read_csv(Config.TEST_CSV)

    test_dataset = BirdDataset(
        test_df, feature_df=feature_df, mode="test", transform=get_transforms("test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
