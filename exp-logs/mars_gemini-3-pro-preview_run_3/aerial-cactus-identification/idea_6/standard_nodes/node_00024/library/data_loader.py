import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config


class CactusDataset(Dataset):
    """
    Custom Dataset for Cactus Identification.
    Handles images loaded as numpy arrays.
    """

    def __init__(self, images, labels=None, ids=None, transform=None):
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, C) numpy array
        img = self.images[idx]

        # Convert to PIL Image for compatibility with torchvision transforms
        img = transforms.ToPILImage()(img)

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label, self.ids[idx]
        else:
            return img, self.ids[idx]


def get_transforms(split="train"):
    """
    Returns the transformation pipeline for a given split.
    """
    # Standardize to [-1, 1] roughly, or use calculated stats.
    # Using 0.5 is standard for simple normalization.
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # Photometric augmentation as per Idea
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )


def _load_raw_data():
    """
    Helper to read images from disk based on metadata CSVs.
    Merges train and validation metadata for CV splitting.
    """
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Combine train and val for Stratified K-Fold CV
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    def load_images_from_df(df):
        imgs = []
        ids = []
        lbls = []
        for _, row in df.iterrows():
            path = os.path.join(Config.INPUT_DIR, row["file_path"])
            img = cv2.imread(path)
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)
                ids.append(row["id"])
                if "has_cactus" in row:
                    lbls.append(row["has_cactus"])

        # Return as numpy arrays
        return np.array(imgs), np.array(lbls), np.array(ids)

    train_imgs, train_lbls, train_ids = load_images_from_df(full_train_df)
    test_imgs, _, test_ids = load_images_from_df(
        test_df
    )  # Ignore test labels (placeholders)

    return train_imgs, train_lbls, train_ids, test_imgs, test_ids


def prepare_data(load_cached_data=True):
    """
    Loads data, handling caching logic.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "train_imgs": os.path.join(cache_dir, "train_imgs.npy"),
        "train_lbls": os.path.join(cache_dir, "train_lbls.npy"),
        "train_ids": os.path.join(cache_dir, "train_ids.npy"),
        "test_imgs": os.path.join(cache_dir, "test_imgs.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data:
        if all(os.path.exists(p) for p in files.values()):
            try:
                train_imgs = np.load(files["train_imgs"])
                train_lbls = np.load(files["train_lbls"])
                train_ids = np.load(files["train_ids"])
                test_imgs = np.load(files["test_imgs"])
                test_ids = np.load(files["test_ids"])
                return train_imgs, train_lbls, train_ids, test_imgs, test_ids
            except Exception:
                # If loading fails, proceed to compute from scratch
                pass

    # 2. IF loading fails OR load_cached_data is False: Compute and Save.
    train_imgs, train_lbls, train_ids, test_imgs, test_ids = _load_raw_data()

    np.save(files["train_imgs"], train_imgs)
    np.save(files["train_lbls"], train_lbls)
    np.save(files["train_ids"], train_ids)
    np.save(files["test_imgs"], test_imgs)
    np.save(files["test_ids"], test_ids)

    return train_imgs, train_lbls, train_ids, test_imgs, test_ids


def get_dataloaders(fold_id=0, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold in Stratified K-Fold CV.
    """
    train_imgs, train_lbls, train_ids, _, _ = prepare_data(load_cached_data)

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the specific fold
    splits = list(skf.split(train_imgs, train_lbls))
    if fold_id >= len(splits):
        raise ValueError(f"Fold {fold_id} out of range for {Config.NUM_FOLDS} folds.")

    train_idx, val_idx = splits[fold_id]

    # Split data
    X_train, X_val = train_imgs[train_idx], train_imgs[val_idx]
    y_train, y_val = train_lbls[train_idx], train_lbls[val_idx]
    ids_train, ids_val = train_ids[train_idx], train_ids[val_idx]

    # Create Datasets
    train_ds = CactusDataset(
        X_train, y_train, ids_train, transform=get_transforms("train")
    )
    val_ds = CactusDataset(X_val, y_val, ids_val, transform=get_transforms("val"))

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for Mixup and BatchNorm stability
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Creates DataLoader for the test set.
    """
    _, _, _, test_imgs, test_ids = prepare_data(load_cached_data)

    test_ds = CactusDataset(
        test_imgs, labels=None, ids=test_ids, transform=get_transforms("test")
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
