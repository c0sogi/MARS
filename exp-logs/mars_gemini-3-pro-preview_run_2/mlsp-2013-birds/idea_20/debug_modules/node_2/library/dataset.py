import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skmultilearn.model_selection import IterativeStratification

from library.config import Config
from library.augmentations import get_transforms


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Handles reading images from memory (numpy array), applying augmentations,
    and returning tensors.
    """

    def __init__(self, images, labels=None, transforms=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, 3).
            labels (np.ndarray, optional): Array of multi-hot labels (N, Num_Classes).
            transforms (A.Compose, optional): Albumentations transformation pipeline.
        """
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image
        image = self.images[idx]

        # Ensure image is uint8 for Albumentations
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback conversion (H, W, C) -> (C, H, W)
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        # Handle Labels
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            # Return dummy labels for test set
            return image, torch.zeros(Config.NUM_SPECIES)


def load_image(path):
    """
    Loads an image from disk, converts to Pseudo-RGB.
    Returns None if file missing or load failed.
    """
    if not os.path.exists(path):
        return None

    # Load as grayscale (spectrograms are typically single channel)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Convert to 3-channel RGB for ImageNet-pretrained models
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def get_data(load_cached_data=True):
    """
    Loads training, validation, and test data.
    Implements caching to disk (parquet/npy) to save time on re-runs.

    Returns:
        df_dev (pd.DataFrame): Combined train+val dataframe with 'fold' column.
        images_dev (np.ndarray): Images for dev set.
        labels_dev (np.ndarray): Labels for dev set.
        df_test (pd.DataFrame): Test dataframe.
        images_test (np.ndarray): Images for test set.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache paths
    dev_imgs_path = os.path.join(cache_dir, "images_dev.npy")
    dev_lbls_path = os.path.join(cache_dir, "labels_dev.npy")
    dev_df_path = os.path.join(cache_dir, "df_dev.parquet")

    test_imgs_path = os.path.join(cache_dir, "images_test.npy")
    test_df_path = os.path.join(cache_dir, "df_test.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(dev_imgs_path)
        and os.path.exists(dev_lbls_path)
        and os.path.exists(dev_df_path)
        and os.path.exists(test_imgs_path)
        and os.path.exists(test_df_path)
    )

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        images_dev = np.load(dev_imgs_path)
        labels_dev = np.load(dev_lbls_path)
        df_dev = pd.read_parquet(dev_df_path)

        images_test = np.load(test_imgs_path)
        df_test = pd.read_parquet(test_df_path)

        return df_dev, images_dev, labels_dev, df_test, images_test

    print("Cache not found or ignored. Processing data from scratch...")

    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Combine train and val into a single development set for K-Fold CV
    df_dev = pd.concat([df_train, df_val], ignore_index=True)

    # Manually shuffle the dataframe to ensure random distribution before splitting.
    # This avoids issues with IterativeStratification not supporting random_state/shuffle correctly.
    df_dev = df_dev.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)

    # Debugging: Reduce dataset size
    if Config.DEBUG:
        print("DEBUG MODE: Reducing dataset size.")
        df_dev = df_dev.head(50).reset_index(drop=True)
        df_test = df_test.head(20).reset_index(drop=True)

    # 2. Process Dev Images
    images_dev = []
    valid_indices_dev = []

    print(f"Processing {len(df_dev)} dev images...")
    for idx, row in df_dev.iterrows():
        rel_path = row["file_path_spec"]

        # Switch to filtered spectrograms if configured
        if Config.USE_FILTERED_SPECTROGRAMS:
            rel_path = rel_path.replace("spectrograms", "filtered_spectrograms")

        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        img = load_image(full_path)

        if img is not None:
            images_dev.append(img)
            valid_indices_dev.append(idx)
        else:
            print(f"Warning: Could not load image {full_path}")

    # Filter dataframe to only valid images
    df_dev = df_dev.iloc[valid_indices_dev].reset_index(drop=True)
    images_dev = np.array(images_dev)

    # 3. Extract Labels
    label_cols = [c for c in df_dev.columns if c.startswith("species_")]
    labels_dev = df_dev[label_cols].values.astype(np.float32)

    # 4. Create Stratified Folds
    print(f"Creating {Config.N_FOLDS} stratified folds...")
    # IterativeStratification expects X and y. We use indices as X.
    X_dummy = df_dev.index.values.reshape(-1, 1)
    y_labels = labels_dev

    # Removed random_state to avoid ValueError with shuffle=False (default).
    # Data is already shuffled above.
    k_fold = IterativeStratification(n_splits=Config.N_FOLDS, order=1)

    df_dev["fold"] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(k_fold.split(X_dummy, y_labels)):
        df_dev.loc[val_idx, "fold"] = fold_idx

    # 5. Process Test Images
    images_test = []
    valid_indices_test = []

    print(f"Processing {len(df_test)} test images...")
    for idx, row in df_test.iterrows():
        rel_path = row["file_path_spec"]
        if Config.USE_FILTERED_SPECTROGRAMS:
            rel_path = rel_path.replace("spectrograms", "filtered_spectrograms")

        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        img = load_image(full_path)

        if img is not None:
            images_test.append(img)
            valid_indices_test.append(idx)
        else:
            print(f"Warning: Could not load image {full_path}")

    df_test = df_test.iloc[valid_indices_test].reset_index(drop=True)
    images_test = np.array(images_test)

    # 6. Save to cache
    print("Saving data to cache...")
    np.save(dev_imgs_path, images_dev)
    np.save(dev_lbls_path, labels_dev)
    df_dev.to_parquet(dev_df_path)

    np.save(test_imgs_path, images_test)
    df_test.to_parquet(test_df_path)

    return df_dev, images_dev, labels_dev, df_test, images_test


def get_dataloaders(
    fold_idx, df_dev, images_dev, labels_dev, batch_size=Config.BATCH_SIZE
):
    """
    Creates train and validation dataloaders for a specific fold.
    """
    # Split data based on fold column
    train_mask = df_dev["fold"] != fold_idx
    val_mask = df_dev["fold"] == fold_idx

    train_imgs = images_dev[train_mask]
    train_lbls = labels_dev[train_mask]

    val_imgs = images_dev[val_mask]
    val_lbls = labels_dev[val_mask]

    # Create Datasets
    train_dataset = BirdDataset(
        images=train_imgs, labels=train_lbls, transforms=get_transforms(mode="train")
    )

    val_dataset = BirdDataset(
        images=val_imgs, labels=val_lbls, transforms=get_transforms(mode="val")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(df_test, images_test, batch_size=Config.BATCH_SIZE):
    """
    Creates the test dataloader.
    """
    test_dataset = BirdDataset(
        images=images_test, labels=None, transforms=get_transforms(mode="test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=False,
    )

    return test_loader
