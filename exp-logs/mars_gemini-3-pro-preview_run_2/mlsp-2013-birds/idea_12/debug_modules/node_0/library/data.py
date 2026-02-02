import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skmultilearn.model_selection import IterativeStratifiedKFold

from library.config import Config
from library.utils import set_seed


class BirdDataset(Dataset):
    """
    Dataset class for loading bird spectrograms and labels.
    Handles pseudo-RGB conversion and resolution-specific transformations.
    """

    def __init__(self, df, transform=None, phase="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata and labels.
            transform (albumentations.Compose): Transformations to apply.
            phase (str): 'train', 'valid', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.phase = phase

        # Identify label columns
        self.label_cols = [c for c in df.columns if c.startswith("species_")]
        self.labels = self.df[self.label_cols].values.astype(np.float32)

        # Pre-compute file paths to avoid overhead in __getitem__
        self.file_paths = []
        for _, row in self.df.iterrows():
            # Use the helper in Config to determine the correct path (filtered vs standard)
            # We pass the file_path_spec from metadata, the helper extracts basename
            fpath = Config.get_spectrogram_path(row["file_path_spec"])
            self.file_paths.append(fpath)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load image (BMP)
        # cv2.imread loads as BGR by default if color, or grayscale if flag is set.
        # Spectrograms are single channel BMPs in the dataset description,
        # but we treat them as images.
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            # Fallback for missing files (should not happen with correct metadata)
            # Create a black image of default size to prevent crash
            img = np.zeros((256, 512), dtype=np.uint8)

        # Ensure image has 3 channels (Pseudo-RGB)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=-1)
        else:
            # If already 3 channels (BGR), convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            # Minimal transform if none provided: Normalize and ToTensor
            base_transform = A.Compose([A.Normalize(), ToTensorV2()])
            augmented = base_transform(image=img)
            img = augmented["image"]

        # Get labels
        label = self.labels[idx]

        # Get rec_id for tracking/submission
        rec_id = self.df.iloc[idx]["rec_id"]

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.float32),
            "rec_id": torch.tensor(rec_id, dtype=torch.long),
        }


def get_transforms(height, width, phase="train"):
    """
    Generates the Albumentations transformation pipeline.

    Args:
        height (int): Target height (Frequency axis).
        width (int): Target width (Time axis).
        phase (str): 'train' or 'valid'/'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    if phase == "train":
        return A.Compose(
            [
                # Resize to target resolution
                A.Resize(height, width),
                # Photometric Augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Geometric / SpecAugment Simulation
                # CoarseDropout masks rectangular regions, simulating time/freq masking
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(height * 0.15),
                    max_width=int(width * 0.15),
                    min_holes=2,
                    min_height=int(height * 0.05),
                    min_width=int(width * 0.05),
                    fill_value=0,
                    p=0.5,
                ),
                # Normalize (ImageNet mean/std is standard for pretrained models)
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height, width),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def prepare_folds(load_cached_data=True):
    """
    Loads training and validation metadata, merges them, and creates
    Iterative Stratified K-Folds. Caches the result to disk.

    Args:
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: DataFrame with 'fold' column.
    """
    cache_path = os.path.join(Config.WORK_DIR, "folds_data.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading folds data from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating new folds data...")

    # 2. Load Metadata
    if not os.path.exists(Config.TRAIN_CSV) or not os.path.exists(Config.VAL_CSV):
        raise FileNotFoundError(
            "Metadata CSVs not found. Run metadata generation first."
        )

    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # 3. Merge Datasets
    # We combine them to perform a fresh K-Fold split on the entire development set
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    # 4. Prepare for Iterative Stratified K-Fold
    # X is just dummy indices, y is the multi-label matrix
    X = full_df.index.values.reshape(-1, 1)

    label_cols = [c for c in full_df.columns if c.startswith("species_")]
    y = full_df[label_cols].values

    # 5. Perform Split
    # Seed skmultilearn via random state if possible, or rely on global seed
    # IterativeStratifiedKFold uses np.random, so set_seed in utils handles it
    k_fold = IterativeStratifiedKFold(
        n_splits=Config.NUM_FOLDS, order=1, random_state=Config.SEED
    )

    full_df["fold"] = -1

    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X, y)):
        full_df.loc[val_indices, "fold"] = fold_idx

    # 6. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    full_df.to_parquet(cache_path, index=False)
    print(f"Saved folds data to {cache_path}")

    return full_df
