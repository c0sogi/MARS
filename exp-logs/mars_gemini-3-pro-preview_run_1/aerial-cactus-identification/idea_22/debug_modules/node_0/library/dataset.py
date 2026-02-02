import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config


def load_dataset_to_memory(load_cached_data=True):
    """
    Loads images and labels into memory. Uses caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from .npy files.

    Returns:
        tuple: (train_imgs, train_labels, test_imgs, test_ids)
            train_imgs: np.ndarray (N, 3, 32, 32) float32
            train_labels: np.ndarray (N,) int
            test_imgs: np.ndarray (M, 3, 32, 32) float32
            test_ids: np.ndarray (M,) string
    """
    cache_exists = (
        os.path.exists(Config.CACHE_TRAIN_IMGS)
        and os.path.exists(Config.CACHE_TRAIN_LABELS)
        and os.path.exists(Config.CACHE_TEST_IMGS)
        and os.path.exists(Config.CACHE_TEST_IDS)
    )

    if load_cached_data and cache_exists:
        print("Loading dataset from cache...")
        train_imgs = np.load(Config.CACHE_TRAIN_IMGS)
        train_labels = np.load(Config.CACHE_TRAIN_LABELS)
        test_imgs = np.load(Config.CACHE_TEST_IMGS)
        test_ids = np.load(Config.CACHE_TEST_IDS)
    else:
        print("Processing dataset from scratch...")

        # 1. Load Metadata
        # Combine train and val metadata to get the full labeled set for CV
        df_train_part = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val_part = pd.read_csv(Config.VAL_METADATA_PATH)
        df_train = pd.concat([df_train_part, df_val_part], ignore_index=True)

        df_test = pd.read_csv(Config.TEST_METADATA_PATH)

        # 2. Helper to load images
        def load_images_from_df(df):
            imgs = []
            ids = []
            labels = []

            # Pre-allocate for speed if desired, but list append is fine for this size
            for _, row in df.iterrows():
                # Metadata file_path is relative to input dir (e.g., "train/id.jpg")
                full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

                img = cv2.imread(full_path)
                if img is None:
                    print(f"Warning: Could not read image {full_path}")
                    continue

                # BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Normalize to 0-1 float32
                img = img.astype(np.float32) / 255.0

                # HWC to CHW
                img = img.transpose(2, 0, 1)

                imgs.append(img)
                ids.append(row["id"])
                if "has_cactus" in row:
                    labels.append(row["has_cactus"])

            return np.array(imgs, dtype=np.float32), np.array(ids), np.array(labels)

        # 3. Process Data
        print(f"Loading {len(df_train)} training images...")
        train_imgs, _, train_labels = load_images_from_df(df_train)
        train_labels = train_labels.astype(
            np.float32
        )  # BCEWithLogitsLoss expects float target

        print(f"Loading {len(df_test)} test images...")
        test_imgs, test_ids, _ = load_images_from_df(df_test)

        # 4. Save to Cache
        print("Saving dataset to cache...")
        np.save(Config.CACHE_TRAIN_IMGS, train_imgs)
        np.save(Config.CACHE_TRAIN_LABELS, train_labels)
        np.save(Config.CACHE_TEST_IMGS, test_imgs)
        np.save(Config.CACHE_TEST_IDS, test_ids)

    # Debug subsetting
    if Config.DEBUG:
        print(f"DEBUG MODE: Truncating dataset to {Config.DEBUG_SUBSET_SIZE} samples.")
        train_imgs = train_imgs[: Config.DEBUG_SUBSET_SIZE]
        train_labels = train_labels[: Config.DEBUG_SUBSET_SIZE]
        test_imgs = test_imgs[: Config.DEBUG_SUBSET_SIZE]
        test_ids = test_ids[: Config.DEBUG_SUBSET_SIZE]

    print(
        f"Dataset Loaded. Train shape: {train_imgs.shape}, Test shape: {test_imgs.shape}"
    )
    return train_imgs, train_labels, test_imgs, test_ids


class CactusDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert numpy array to tensor
        img = torch.from_numpy(self.images[idx])

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label

        return img


def get_transforms(mode="train"):
    """
    Returns the transformations for the specified mode.

    Stats derived from dataset analysis:
    Mean: [0.503, 0.452, 0.468]
    Std:  [0.151, 0.140, 0.154]
    """
    # Dataset specific statistics
    mean = [0.503, 0.452, 0.468]
    std = [0.151, 0.140, 0.154]

    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        return transforms.Compose([transforms.Normalize(mean=mean, std=std)])


def get_fold_loaders(
    fold_idx, data, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates train and validation loaders for a specific fold using StratifiedKFold.

    Args:
        fold_idx (int): The current fold index (0 to N_FOLDS-1).
        data (tuple): (train_imgs, train_labels)
        batch_size (int): Batch size.
        num_workers (int): Number of workers.

    Returns:
        train_loader, val_loader
    """
    train_imgs, train_labels = data

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    # We iterate through the generator to find the specific fold indices
    splits = list(skf.split(train_imgs, train_labels))
    train_idx, val_idx = splits[fold_idx]

    # Subset data
    X_train, y_train = train_imgs[train_idx], train_labels[train_idx]
    X_val, y_val = train_imgs[val_idx], train_labels[val_idx]

    # Create Datasets
    train_dataset = CactusDataset(
        X_train, y_train, transform=get_transforms(mode="train")
    )
    val_dataset = CactusDataset(X_val, y_val, transform=get_transforms(mode="valid"))

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
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
    test_imgs, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates a loader for the test set.
    """
    test_dataset = CactusDataset(
        test_imgs, labels=None, transform=get_transforms(mode="valid")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
