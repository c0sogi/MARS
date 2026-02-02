import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_CSV,
    VAL_METADATA_CSV,
    TEST_METADATA_CSV,
    CACHE_TRAIN_IMGS,
    CACHE_TRAIN_LABELS,
    CACHE_TRAIN_IDS,
    CACHE_TEST_IMGS,
    CACHE_TEST_IDS,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    N_FOLDS,
    CACHE_DIR,
)
from library.utils import seed_everything

# Dataset specific statistics calculated from analysis
# Mean: R=128.37, G=115.25, B=119.40 -> /255
# Std: R=38.60, G=35.68, B=39.15 -> /255
MEAN = (0.503, 0.452, 0.468)
STD = (0.151, 0.140, 0.154)


def get_transforms(phase="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        phase (str): 'train', 'valid', or 'test'.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )


class CactusDataset(Dataset):
    """
    Custom Dataset for Cactus Identification using in-memory numpy arrays.
    """

    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray, optional): Array of labels (N,).
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image from memory
        image = self.images[idx]

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (shouldn't happen in pipeline)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            return image


def load_dataset_arrays(load_cached_data=True, is_test=False):
    """
    Loads dataset into memory. Uses caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from .npy files.
        is_test (bool): If True, loads test data. Else loads full training data.

    Returns:
        tuple: (images, labels, ids) for train, (images, ids) for test.
    """
    # Define paths based on mode
    if is_test:
        path_imgs = CACHE_TEST_IMGS
        path_ids = CACHE_TEST_IDS
        # Test set doesn't have ground truth labels we can use for training
        path_labels = None
    else:
        path_imgs = CACHE_TRAIN_IMGS
        path_labels = CACHE_TRAIN_LABELS
        path_ids = CACHE_TRAIN_IDS

    # 1. Try to load from cache
    if load_cached_data:
        if is_test:
            if os.path.exists(path_imgs) and os.path.exists(path_ids):
                print(f"Loading cached test data from {CACHE_DIR}...")
                imgs = np.load(path_imgs)
                ids = np.load(path_ids)
                return imgs, ids
        else:
            if (
                os.path.exists(path_imgs)
                and os.path.exists(path_labels)
                and os.path.exists(path_ids)
            ):
                print(f"Loading cached training data from {CACHE_DIR}...")
                imgs = np.load(path_imgs)
                labels = np.load(path_labels)
                ids = np.load(path_ids)
                return imgs, labels, ids

    # 2. Process from scratch if cache missing or load_cached_data=False
    print(f"Processing {'test' if is_test else 'training'} data from scratch...")

    if is_test:
        df = pd.read_csv(TEST_METADATA_CSV)
    else:
        # Combine train and val metadata to get the full labeled dataset for CV
        df_train = pd.read_csv(TRAIN_METADATA_CSV)
        df_val = pd.read_csv(VAL_METADATA_CSV)
        df = pd.concat([df_train, df_val], ignore_index=True)

    img_list = []
    id_list = []
    label_list = []

    # Pre-allocate for efficiency could be done, but list append is acceptable for this size
    for _, row in df.iterrows():
        # file_path in metadata is relative to INPUT_DIR (e.g., "train/xxx.jpg")
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        img = cv2.imread(full_path)
        if img is None:
            print(f"Warning: Could not read image {full_path}")
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_list.append(img)
        id_list.append(row["id"])

        if not is_test:
            label_list.append(row["has_cactus"])

    # Convert to numpy arrays
    imgs_np = np.array(img_list, dtype=np.uint8)
    ids_np = np.array(id_list)

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Save to cache
    np.save(path_imgs, imgs_np)
    np.save(path_ids, ids_np)

    if not is_test:
        labels_np = np.array(label_list, dtype=np.float32)
        np.save(path_labels, labels_np)
        print(f"Cached {len(imgs_np)} training samples.")
        return imgs_np, labels_np, ids_np
    else:
        print(f"Cached {len(imgs_np)} test samples.")
        return imgs_np, ids_np


def get_fold_dataloaders(fold_idx, load_cached_data=True):
    """
    Generates train and validation DataLoaders for a specific fold.

    Args:
        fold_idx (int): The fold index (0 to N_FOLDS-1).
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        train_loader, val_loader
    """
    seed_everything(SEED)

    # Load full dataset
    images, labels, ids = load_dataset_arrays(
        load_cached_data=load_cached_data, is_test=False
    )

    # Create Stratified K-Fold Split
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Get indices for the requested fold
    # skf.split requires X and y, though X can be zeros
    splits = list(skf.split(images, labels))
    train_idx, val_idx = splits[fold_idx]

    # Subset data
    X_train, y_train = images[train_idx], labels[train_idx]
    X_val, y_val = images[val_idx], labels[val_idx]

    # Create Datasets
    train_dataset = CactusDataset(
        images=X_train, labels=y_train, transform=get_transforms(phase="train")
    )

    val_dataset = CactusDataset(
        images=X_val, labels=y_val, transform=get_transforms(phase="valid")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for Mixup/BatchNorm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Generates the Test DataLoader.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        test_loader, test_ids
    """
    images, ids = load_dataset_arrays(load_cached_data=load_cached_data, is_test=True)

    test_dataset = CactusDataset(
        images=images, labels=None, transform=get_transforms(phase="test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, ids
