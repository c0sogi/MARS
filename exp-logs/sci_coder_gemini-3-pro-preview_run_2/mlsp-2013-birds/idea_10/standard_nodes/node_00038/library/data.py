import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from skmultilearn.model_selection import IterativeStratification
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline for the specified mode.

    Args:
        mode (str): 'train' or 'val'/'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    height, width = Config.IMAGE_SIZE

    if mode == "train":
        return A.Compose(
            [
                # Rectangular Resolution
                A.Resize(height=height, width=width),
                # Photometric Augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Geometric / SpecAugment Simulation
                # CoarseDropout simulates time/frequency masking
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(height * 0.1),
                    max_width=int(width * 0.1),
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                # Normalization (ImageNet stats)
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=height, width=width),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class BirdDataset(Dataset):
    """
    Custom Dataset for Bird Species Classification.
    Loads spectrograms, applies Pseudo-RGB conversion, and augmentations.
    """

    def __init__(self, df, transforms=None, root_dir=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transforms (A.Compose): Albumentations transforms.
            root_dir (str): Directory containing the filtered spectrograms.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.root_dir = root_dir if root_dir else Config.FILTERED_SPECTROGRAM_DIR

        # Identify label columns
        self.label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

        # Pre-check file existence to avoid runtime errors, or handle in __getitem__
        # For efficiency, we assume paths are mostly correct but handle errors in getitem

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["rec_id"]

        # Extract filename from the metadata path and construct new path
        # Metadata path example: supplemental_data/spectrograms/PC10_....bmp
        # We need to point to Config.FILTERED_SPECTROGRAM_DIR
        original_rel_path = row["file_path_spec"]
        filename = os.path.basename(original_rel_path)
        file_path = os.path.join(self.root_dir, filename)

        # Load Image
        image = None
        if os.path.exists(file_path):
            # Read as grayscale
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Handle missing or corrupt files
        if image is None:
            # Create a black dummy image of target size (before resize)
            # Default spectrogram size is roughly 256x1000+, we make a placeholder
            image = np.zeros(
                (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]), dtype=np.uint8
            )

        # Pseudo-RGB: Replicate channels
        # Shape becomes (H, W, 3)
        image = cv2.merge([image, image, image])

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion
            image = ToTensorV2()(image=image)["image"]

        # Get Labels
        labels = row[self.label_cols].values.astype(np.float32)

        return image, torch.tensor(labels), torch.tensor(rec_id, dtype=torch.long)


def process_folds(load_cached_data=True):
    """
    Loads training and validation metadata, combines them, and generates
    stratified K-Folds. Caches the result to disk.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Combined dataframe with a 'fold' column.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "folds_data.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # print(f"Loaded folds from cache: {cache_path}")
            return df
        except Exception as e:
            pass
            # print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load and Combine Data
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine (Fold 0 data)
    df = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # 3. Generate Folds using Iterative Stratification
    # This handles the multi-label nature of the data
    X = df["rec_id"].values.reshape(-1, 1)
    label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]
    y = df[label_cols].values

    mskf = IterativeStratification(n_splits=Config.N_FOLDS, order=1)

    df["fold"] = -1

    for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
        df.loc[val_idx, "fold"] = fold

    # 4. Cache the result
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_loaders(
    fold,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
):
    """
    Creates DataLoaders for a specific fold.

    Args:
        fold (int): The fold index to use for validation (0 to N_FOLDS-1).
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        train_loader, val_loader
    """
    df = process_folds(load_cached_data=True)

    if debug:
        df = df.sample(
            n=min(len(df), Config.DEBUG_SUBSET_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Split based on fold
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    # Create Datasets
    train_dataset = BirdDataset(
        train_df,
        transforms=get_transforms(mode="train"),
        root_dir=Config.FILTERED_SPECTROGRAM_DIR,
    )

    val_dataset = BirdDataset(
        val_df,
        transforms=get_transforms(mode="val"),
        root_dir=Config.FILTERED_SPECTROGRAM_DIR,
    )

    # Create DataLoaders
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


def get_test_loader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates a DataLoader for the test set (Fold 1).

    Returns:
        test_loader
    """
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    test_dataset = BirdDataset(
        df_test,
        transforms=get_transforms(mode="test"),
        root_dir=Config.FILTERED_SPECTROGRAM_DIR,
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
