import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything

# ------------------------------------------------------------------------------
# 1. Augmentation / Transforms
# ------------------------------------------------------------------------------


def get_transforms(phase="train"):
    """
    Returns the data transformation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # Standard ImageNet normalization stats
    # Although dataset is aerial cactus, using ImageNet stats is standard for
    # transfer learning and generally safe for custom models too.
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    if phase == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # Photometric augmentation as per Idea
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ]
        )
    else:
        # Validation / Test: Only normalization
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ]
        )


# ------------------------------------------------------------------------------
# 2. Dataset Class
# ------------------------------------------------------------------------------


class CactusDataset(Dataset):
    """
    PyTorch Dataset for the Cactus Identification task.
    """

    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray, optional): Array of labels with shape (N,).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image (already loaded in memory as numpy array)
        img = self.images[idx]

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            label = self.labels[idx]
            # Return label as float for BCEWithLogitsLoss
            return img, torch.tensor(label, dtype=torch.float32)
        else:
            return img


# ------------------------------------------------------------------------------
# 3. Data Preparation & Caching
# ------------------------------------------------------------------------------


def _load_raw_images(df, input_dir):
    """
    Helper to load images listed in a dataframe.
    """
    images = []
    ids = []
    labels = []

    # Pre-allocate if possible or just append (dataset is small enough ~17k)
    # Appending is fine for this size.

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            # Should not happen given metadata verification, but good for safety
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img)
        ids.append(row["id"])
        if "has_cactus" in row:
            labels.append(row["has_cactus"])

    return np.array(images), np.array(labels), np.array(ids)


def _get_data_arrays(load_cached_data=True):
    """
    Loads dataset into memory. Uses caching to speed up subsequent runs.
    Combines train and val metadata to form the full training set for CV.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_imgs, train_lbls, test_imgs, test_ids)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_train_imgs = os.path.join(cache_dir, "full_train_imgs.npy")
    cache_train_lbls = os.path.join(cache_dir, "full_train_lbls.npy")
    cache_test_imgs = os.path.join(cache_dir, "test_imgs.npy")
    cache_test_ids = os.path.join(cache_dir, "test_ids.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(cache_train_imgs)
        and os.path.exists(cache_train_lbls)
        and os.path.exists(cache_test_imgs)
        and os.path.exists(cache_test_ids)
    )

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        train_imgs = np.load(cache_train_imgs)
        train_lbls = np.load(cache_train_lbls)
        test_imgs = np.load(cache_test_imgs)
        test_ids = np.load(cache_test_ids)
    else:
        print("Processing raw data...")
        # Load metadata
        df_train_part = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val_part = pd.read_csv(Config.VAL_METADATA_PATH)
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)

        # Combine train and val parts to get full training set for CV
        df_full_train = pd.concat([df_train_part, df_val_part], ignore_index=True)

        # Debug mode: subsample
        if Config.DEBUG:
            df_full_train = df_full_train.sample(
                n=500, random_state=Config.SEED
            ).reset_index(drop=True)
            df_test = df_test.sample(n=100, random_state=Config.SEED).reset_index(
                drop=True
            )

        # Load images
        train_imgs, train_lbls, _ = _load_raw_images(df_full_train, Config.INPUT_DIR)
        test_imgs, _, test_ids = _load_raw_images(df_test, Config.INPUT_DIR)

        # Save to cache
        np.save(cache_train_imgs, train_imgs)
        np.save(cache_train_lbls, train_lbls)
        np.save(cache_test_imgs, test_imgs)
        np.save(cache_test_ids, test_ids)

    return train_imgs, train_lbls, test_imgs, test_ids


# ------------------------------------------------------------------------------
# 4. Public Loader Functions
# ------------------------------------------------------------------------------


def get_loaders(fold_idx, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold in Stratified K-Fold Cross Validation.

    Args:
        fold_idx (int): The index of the fold to use for validation (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (train_loader, val_loader)
    """
    seed_everything(Config.SEED)

    # Load all data
    train_imgs, train_lbls, _, _ = _get_data_arrays(load_cached_data=load_cached_data)

    # Create Stratified K-Fold split
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    # skf.split returns a generator, we iterate to find the specific fold
    for i, (train_index, val_index) in enumerate(skf.split(train_imgs, train_lbls)):
        if i == fold_idx:
            # Split data
            X_train, X_val = train_imgs[train_index], train_imgs[val_index]
            y_train, y_val = train_lbls[train_index], train_lbls[val_index]

            # Create Datasets
            train_dataset = CactusDataset(
                X_train, y_train, transform=get_transforms("train")
            )
            val_dataset = CactusDataset(X_val, y_val, transform=get_transforms("val"))

            # Create DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,  # Useful for Mixup/BatchNorm stability
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            return train_loader, val_loader

    raise ValueError(f"Fold index {fold_idx} out of range (0-{Config.NUM_FOLDS-1})")


def get_test_loader(load_cached_data=True):
    """
    Creates a DataLoader for the test set.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (test_loader, test_ids)
    """
    # Load data
    _, _, test_imgs, test_ids = _get_data_arrays(load_cached_data=load_cached_data)

    # Create Dataset
    test_dataset = CactusDataset(
        test_imgs, labels=None, transform=get_transforms("test")
    )

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, test_ids
