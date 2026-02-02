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
from library.utils import worker_init_fn


class BirdDataset(Dataset):
    """
    Custom Dataset for loading Bird Spectrograms.
    Handles Pseudo-RGB conversion and multi-label targets.
    """

    def __init__(self, df, phase, resolution, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            phase (str): 'train', 'val', or 'test'.
            resolution (tuple): Target resolution (height, width).
            transform (A.Compose): Albumentations transforms pipeline.
        """
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.resolution = resolution
        self.transform = transform

        # Identify label columns (species_0 ... species_18)
        self.label_cols = [c for c in df.columns if c.startswith("species_")]
        self.labels = self.df[self.label_cols].values.astype(np.float32)

        # Pre-calculate full file paths
        # Metadata contains relative paths, e.g., "supplemental_data/..."
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, rel_path)
            for rel_path in self.df["file_path_spec"]
        ]

        # Store rec_ids for test phase
        self.rec_ids = self.df["rec_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load image (Spectrograms are grayscale BMPs)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Robustness: Handle missing files by creating a blank image
            # Default size approx 256x1246 based on EDA
            img = np.zeros((256, 1246), dtype=np.uint8)

        # Convert to Pseudo-RGB (3 channels) for ImageNet models
        img = cv2.merge([img, img, img])

        # Apply Transforms (Augmentation -> Resize -> Normalize -> Tensor)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # Get Label
        label = self.labels[idx]

        if self.phase == "test":
            # Return rec_id for submission mapping
            rec_id = self.rec_ids[idx]
            return img, label, rec_id
        else:
            return img, label


def get_transforms(phase, resolution):
    """
    Constructs the augmentation pipeline.

    Args:
        phase (str): 'train' or 'val'/'test'.
        resolution (tuple): (height, width).
    """
    height, width = resolution

    # ImageNet Normalization Constants
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=height, width=width),
                # Photometric Augmentation
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Geometric / SpecAugment Simulation
                # CoarseDropout masks out rectangular regions (time/freq masking)
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(height * 0.15),
                    max_width=int(width * 0.15),
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test: Deterministic resizing and normalization
        return A.Compose(
            [
                A.Resize(height=height, width=width),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_fold_dataloaders(fold_idx, model_name, batch_size=None):
    """
    Prepares DataLoaders for a specific fold using Iterative Stratification.
    Combines train.csv and val.csv to perform dynamic K-Fold splitting.

    Args:
        fold_idx (int): Fold index (0 to 4).
        model_name (str): Key from Config.MODEL_SPECS.
        batch_size (int, optional): Batch size override.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Retrieve model-specific resolution
    if model_name not in Config.MODEL_SPECS:
        raise ValueError(f"Model {model_name} not found in Config.MODEL_SPECS")
    resolution = Config.MODEL_SPECS[model_name]["resolution"]

    # Load and Combine Metadata
    df_train_part = pd.read_csv(Config.TRAIN_CSV)
    df_val_part = pd.read_csv(Config.VAL_CSV)
    df_dev = pd.concat([df_train_part, df_val_part], ignore_index=True)

    # Debugging: Subsample data
    if Config.DEBUG:
        df_dev = df_dev.sample(
            n=min(len(df_dev), 40), random_state=Config.SEED
        ).reset_index(drop=True)

    # Prepare for Splitting
    X = df_dev["rec_id"].values.reshape(-1, 1)
    label_cols = [c for c in df_dev.columns if c.startswith("species_")]
    y = df_dev[label_cols].values

    # Perform Iterative Stratified K-Fold
    # This ensures balanced multi-label distribution across folds
    k_fold = IterativeStratification(
        n_splits=Config.NUM_FOLDS, order=1, random_state=Config.SEED
    )

    # Generate splits
    # Note: split() returns a generator
    splits = list(k_fold.split(X, y))

    if fold_idx < 0 or fold_idx >= len(splits):
        raise ValueError(f"Fold index {fold_idx} out of range.")

    train_indices, val_indices = splits[fold_idx]

    df_train_fold = df_dev.iloc[train_indices].reset_index(drop=True)
    df_val_fold = df_dev.iloc[val_indices].reset_index(drop=True)

    # Create Datasets
    train_dataset = BirdDataset(
        df_train_fold,
        phase="train",
        resolution=resolution,
        transform=get_transforms("train", resolution),
    )

    val_dataset = BirdDataset(
        df_val_fold,
        phase="val",
        resolution=resolution,
        transform=get_transforms("val", resolution),
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,  # Important for BatchNorm stability in small batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader


def get_test_dataloader(model_name, batch_size=None):
    """
    Prepares DataLoader for the test set.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    if model_name not in Config.MODEL_SPECS:
        raise ValueError(f"Model {model_name} not found in Config.MODEL_SPECS")
    resolution = Config.MODEL_SPECS[model_name]["resolution"]

    df_test = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        df_test = df_test.iloc[:20].reset_index(drop=True)

    test_dataset = BirdDataset(
        df_test,
        phase="test",
        resolution=resolution,
        transform=get_transforms("test", resolution),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return test_loader
