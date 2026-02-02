import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import (
    PLANES,
    BATCH_SIZE,
    N_FOLDS,
    SEED,
    get_transforms,
    METADATA_DIR,
)
from library.preprocessing import process_dataset

# Mapping corresponds to the stacking order in preprocessing.py:
# [expert_dict["lower"], expert_dict["center"], expert_dict["upper"]]
PLANE_TO_IDX = {"lower": 0, "center": 1, "upper": 2}


class RASSEDataset(Dataset):
    """
    PyTorch Dataset for the ROI-Adaptive Spatially-Stratified Ensemble.
    Fetches the specific 2D plane (Lower, Center, or Upper) from the pre-processed
    subject stack and applies Albumentations transforms.
    """

    def __init__(self, images, ids, labels=None, plane_name="center", transform=None):
        """
        Args:
            images (np.ndarray): Array of shape (N, 3, H, W, 3).
            ids (np.ndarray): Array of shape (N,).
            labels (np.ndarray, optional): Array of shape (N,). Defaults to None.
            plane_name (str): One of 'lower', 'center', 'upper'.
            transform (albumentations.Compose): Transforms to apply.
        """
        self.images = images
        self.ids = ids
        self.labels = labels
        self.transform = transform

        if plane_name not in PLANE_TO_IDX:
            raise ValueError(
                f"Invalid plane_name '{plane_name}'. Must be one of {list(PLANE_TO_IDX.keys())}"
            )
        self.plane_idx = PLANE_TO_IDX[plane_name]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve the stack for this subject: (3, H, W, 3)
        # Select the specific expert plane: (H, W, 3)
        img = self.images[idx, self.plane_idx]

        # Apply transforms
        if self.transform:
            # Albumentations expects 'image' keyword
            augmented = self.transform(image=img)
            img_tensor = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        item = {"image": img_tensor, "BraTS21ID": self.ids[idx]}

        if self.labels is not None:
            # Return float tensor for BCEWithLogitsLoss
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


def get_fold_dataloaders(fold_idx, plane_name, batch_size=BATCH_SIZE, num_workers=2):
    """
    Generates Train and Validation DataLoaders for a specific fold and expert plane.
    Uses ONLY the training metadata for Cross-Validation to prevent leakage into the
    hold-out validation set.
    """
    # 1. Load Metadata
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")

    df_train = pd.read_csv(train_meta_path)

    # 2. Process Data (Cached)
    # Changed save_name to "train" to reflect that this is only the training set
    images, ids, labels = process_dataset(
        df_train, load_cached_data=True, save_name="train"
    )

    if labels is None:
        raise ValueError("Labels are missing for the training dataset.")

    # 3. Create Folds
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Get indices for the requested fold
    # We iterate to find the specific fold indices
    fold_generator = skf.split(images, labels)
    train_indices, val_indices = next(
        x for i, x in enumerate(fold_generator) if i == fold_idx
    )

    # 4. Slice Data
    X_train = images[train_indices]
    y_train = labels[train_indices]
    ids_train = ids[train_indices]

    X_val = images[val_indices]
    y_val = labels[val_indices]
    ids_val = ids[val_indices]

    # 5. Create Datasets
    train_dataset = RASSEDataset(
        X_train,
        ids_train,
        y_train,
        plane_name=plane_name,
        transform=get_transforms(phase="train"),
    )

    val_dataset = RASSEDataset(
        X_val,
        ids_val,
        y_val,
        plane_name=plane_name,
        transform=get_transforms(phase="val"),
    )

    # 6. Create DataLoaders
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


def get_test_dataloader(plane_name, batch_size=BATCH_SIZE, num_workers=2):
    """
    Generates the Test DataLoader for a specific expert plane.
    """
    # 1. Load Metadata
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")
    df_test = pd.read_csv(test_meta_path)

    # 2. Process Data (Cached)
    images, ids, _ = process_dataset(df_test, load_cached_data=True, save_name="test")

    # 3. Create Dataset
    test_dataset = RASSEDataset(
        images,
        ids,
        labels=None,
        plane_name=plane_name,
        transform=get_transforms(phase="test"),
    )

    # 4. Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
