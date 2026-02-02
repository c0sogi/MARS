import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skmultilearn.model_selection import IterativeStratification

from library.config import Config
from library.utils import seed_everything


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Loads filtered spectrograms, applies rectangular resizing, pseudo-RGB conversion,
    and augmentations.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (filenames, labels).
            transforms (albumentations.Compose): Albumentations pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        # Identify label columns
        self.label_cols = [c for c in df.columns if c.startswith("species_")]
        self.labels = self.df[self.label_cols].values.astype(np.float32)

        # Pre-compute absolute paths
        self.file_paths = [
            Config.get_spectrogram_path(f) for f in self.df["file_path_spec"]
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load image (BMP is typically single channel or BGR)
        # We load unchanged to inspect channels, but usually these are grayscale
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            # Fallback for missing files (should not happen with validated metadata)
            # Create a black image of correct size
            image = np.zeros(
                (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]), dtype=np.uint8
            )

        # Ensure image is 2D (grayscale) before processing
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to Rectangular Resolution: (Width=448, Height=224)
        # cv2.resize expects (width, height)
        target_h, target_w = Config.IMAGE_SIZE
        image = cv2.resize(image, (target_w, target_h))

        # Convert to Pseudo-RGB (3 channels)
        image = np.stack([image, image, image], axis=-1)

        # Apply Augmentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode == "test":
            # Return image and ID for submission generation
            rec_id = self.df.iloc[idx]["rec_id"]
            return image, rec_id
        else:
            # Return image and multi-hot labels
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): 'train' or 'val'.
    """
    target_h, target_w = Config.IMAGE_SIZE

    # ImageNet normalization stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if mode == "train":
        return A.Compose(
            [
                # Photometric Augmentation
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # SpecAugment Simulation (Time/Freq Masking) via CoarseDropout
                # We use rectangular holes to simulate masking time steps or frequency bands
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(target_h * 0.2),  # Max 20% of freq bands
                    max_width=int(target_w * 0.1),  # Max 10% of time steps
                    min_holes=2,
                    fill_value=0,
                    p=0.5,
                ),
                # Normalize and Convert to Tensor
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def make_folds(load_cached_data=True):
    """
    Loads training data, merges train/val splits, and creates 5 stratified folds.
    Caches the result to parquet.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        pd.DataFrame: Dataframe with a 'fold' column.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "folds_data.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached folds from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating new folds...")
    # Load original metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Combine to form the full development set
    df = pd.concat([train_df, val_df], ignore_index=True)

    # Prepare for Iterative Stratification
    # We need X (indices) and y (labels)
    X = df.index.values.reshape(-1, 1)
    label_cols = [c for c in df.columns if c.startswith("species_")]
    y = df[label_cols].values

    # Initialize folds
    df["fold"] = -1

    # Seed skmultilearn's random state if possible, or rely on global numpy seed
    # IterativeStratifiedKFold doesn't accept a random_state directly in older versions,
    # but relies on numpy.random.
    seed_everything(Config.SEED)

    k_fold = IterativeStratification(n_splits=Config.NUM_FOLDS, order=1)

    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X, y)):
        df.loc[val_indices, "fold"] = fold_idx

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(fold_idx, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold.

    Args:
        fold_idx (int): The fold index to use for validation (0-4).
        load_cached_data (bool): Whether to use cached fold definitions.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get dataframe with fold assignments
    df = make_folds(load_cached_data=load_cached_data)

    # Split into train and val for this fold
    train_df = df[df["fold"] != fold_idx].copy()
    val_df = df[df["fold"] == fold_idx].copy()

    # Debug mode: reduce dataset size
    if Config.DEBUG:
        train_df = train_df.sample(n=32, random_state=Config.SEED).reset_index(
            drop=True
        )
        val_df = val_df.sample(n=16, random_state=Config.SEED).reset_index(drop=True)

    # Create Datasets
    train_dataset = BirdDataset(
        train_df, transforms=get_transforms(mode="train"), mode="train"
    )
    val_dataset = BirdDataset(val_df, transforms=get_transforms(mode="val"), mode="val")

    # Worker Init Function for Reproducibility (Cite Lesson 18)
    def worker_init_fn(worker_id):
        # Use torch.initial_seed() which is generated by the main process
        # and is unique per epoch/worker combination if properly handled by PyTorch
        seed = (torch.initial_seed() + worker_id) % (2**32)
        np.random.seed(seed)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,  # Drop incomplete batches to maintain stats stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader


def get_test_dataloader():
    """
    Creates a DataLoader for the test set.
    """
    df_test = pd.read_csv(Config.TEST_CSV)

    test_dataset = BirdDataset(
        df_test,
        transforms=get_transforms(mode="val"),  # No augmentation for test
        mode="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
