import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from library.config import CFG


class AppleDataset(Dataset):
    """
    Custom Dataset for Apple Disease Detection.
    Loads images, applies transformations, and returns image tensors and labels.
    """

    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test

        # Pre-compute full file paths
        # CFG.input_dir is "./input", file_path in df is relative e.g. "images/Test_0.jpg"
        self.file_paths = [
            os.path.join(CFG.input_dir, fp) for fp in df["file_path"].values
        ]

        if not self.is_test:
            # Extract labels in the order defined in CFG.class_labels
            # Labels are binary indicators (0 or 1)
            self.labels = df[CFG.class_labels].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {path}")

        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Albumentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.is_test:
            # Return dummy label for test set
            return image, torch.tensor(0.0)
        else:
            label = torch.tensor(self.labels[idx])
            return image, label


def get_transforms(data, img_size):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): 'train', 'valid', or 'test'.
        img_size (int): Target image size (e.g., 380 for B4, 224 for Tiny).
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                # Strong Geometric Augmentation
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                # Strictly excluding occlusion augmentations (Cutout, CoarseDropout)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def prepare_loaders(fold, backbone, debug=False):
    """
    Prepares DataLoaders for a specific fold and backbone using Stratified K-Fold.

    Args:
        fold (int): The fold index (0 to n_folds-1).
        backbone (str): The model backbone name to determine image size.
        debug (bool): If True, uses a small subset of data for debugging.

    Returns:
        train_loader, val_loader
    """
    # Load metadata
    train_meta = pd.read_csv(CFG.train_csv)
    val_meta = pd.read_csv(CFG.val_csv)

    # Combine train and val metadata to perform full K-Fold on the entire labeled dataset
    df = pd.concat([train_meta, val_meta], axis=0).reset_index(drop=True)

    # Debug mode: sample subset
    if debug:
        df = df.sample(
            n=min(CFG.debug_sample_size, len(df)), random_state=CFG.seed
        ).reset_index(drop=True)

    # Perform Stratified K-Fold
    # We use the 'stratify_label' column which represents the single active class
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    # Generate fold indices
    folds = list(skf.split(df, df[CFG.target_col]))

    if fold >= len(folds):
        raise ValueError(f"Fold {fold} out of range for {CFG.n_folds} folds.")

    train_idx, val_idx = folds[fold]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Determine image size based on backbone
    if backbone not in CFG.img_sizes:
        raise ValueError(f"Backbone {backbone} not found in CFG.img_sizes.")
    img_size = CFG.img_sizes[backbone]

    # Create Datasets
    train_dataset = AppleDataset(
        train_df, transform=get_transforms("train", img_size), is_test=False
    )
    val_dataset = AppleDataset(
        val_df, transform=get_transforms("valid", img_size), is_test=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def prepare_test_loader(backbone):
    """
    Prepares the DataLoader for the test set.

    Args:
        backbone (str): The model backbone name to determine image size.

    Returns:
        test_loader, image_ids
    """
    test_df = pd.read_csv(CFG.test_csv)

    if backbone not in CFG.img_sizes:
        raise ValueError(f"Backbone {backbone} not found in CFG.img_sizes.")
    img_size = CFG.img_sizes[backbone]

    test_dataset = AppleDataset(
        test_df, transform=get_transforms("test", img_size), is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    return test_loader, test_df["image_id"].values
