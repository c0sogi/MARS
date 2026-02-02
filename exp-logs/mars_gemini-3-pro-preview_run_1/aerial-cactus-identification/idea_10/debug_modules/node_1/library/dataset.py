import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


def _load_raw_data(df, input_dir):
    """
    Reads images from disk, converts to RGB, and extracts file sizes.
    """
    img_paths = df["file_path"].apply(lambda x: os.path.join(input_dir, x)).values

    imgs = []
    file_sizes = []

    # Pre-allocate if possible or just list append (efficient enough for 17k images)
    for path in img_paths:
        # Read image
        img = cv2.imread(path)
        if img is None:
            # Fallback for missing images (should not happen based on metadata check)
            img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
            f_size = 0
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            f_size = os.path.getsize(path)

        imgs.append(img)
        file_sizes.append(f_size)

    return np.array(imgs), np.array(file_sizes)


def get_data(mode="train", load_cached_data=True):
    """
    Retrieves data from cache or processes it from scratch.

    Args:
        mode (str): 'train' (combines train+val metadata) or 'test'.
        load_cached_data (bool): Whether to attempt loading from .npy files.

    Returns:
        tuple: (images, labels, file_sizes, ids)
               ids is only returned for test mode, labels is None for test mode.
    """
    # Define cache paths based on mode
    if mode == "train":
        cache_imgs = Config.CACHE_FULL_TRAIN_IMGS
        cache_labels = Config.CACHE_FULL_TRAIN_LABELS
        cache_fs = Config.CACHE_FULL_TRAIN_FILESIZES
        # No ID cache needed for train usually, but we keep indices consistent
    else:
        cache_imgs = Config.CACHE_TEST_IMGS
        cache_ids = Config.CACHE_TEST_IDS
        cache_fs = Config.CACHE_TEST_FILESIZES
        cache_labels = None

    # Attempt to load cache
    if load_cached_data:
        if mode == "train":
            if (
                os.path.exists(cache_imgs)
                and os.path.exists(cache_labels)
                and os.path.exists(cache_fs)
            ):
                print(f"Loading cached {mode} data from {Config.WORKING_DIR}...")
                imgs = np.load(cache_imgs)
                labels = np.load(cache_labels)
                file_sizes = np.load(cache_fs)
                return imgs, labels, file_sizes
        else:
            if (
                os.path.exists(cache_imgs)
                and os.path.exists(cache_ids)
                and os.path.exists(cache_fs)
            ):
                print(f"Loading cached {mode} data from {Config.WORKING_DIR}...")
                imgs = np.load(cache_imgs)
                ids = np.load(cache_ids)
                file_sizes = np.load(cache_fs)
                return imgs, ids, file_sizes

    print(f"Processing {mode} data from scratch...")

    if mode == "train":
        # Merge train and val metadata for 5-Fold CV
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        df = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

        imgs, file_sizes = _load_raw_data(df, Config.INPUT_DIR)
        labels = df["has_cactus"].values.astype(np.float32)

        # Save cache
        np.save(cache_imgs, imgs)
        np.save(cache_labels, labels)
        np.save(cache_fs, file_sizes)

        return imgs, labels, file_sizes

    else:
        df = pd.read_csv(Config.TEST_METADATA_PATH)
        imgs, file_sizes = _load_raw_data(df, Config.INPUT_DIR)
        ids = df["id"].values

        # Save cache
        np.save(cache_imgs, imgs)
        np.save(cache_ids, ids)
        np.save(cache_fs, file_sizes)

        return imgs, ids, file_sizes


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class CactusDataset(Dataset):
    def __init__(self, images, file_sizes, labels=None, transforms=None):
        self.images = images
        self.file_sizes = file_sizes
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        file_size = self.file_sizes[idx]

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Normalize file size roughly to keep it in a reasonable range for NN/LogisticReg
        # Using a fixed scalar based on analysis (mean ~1000, std ~500)
        # This is a simple robust scaling: (x - 1000) / 500
        norm_file_size = (float(file_size) - 1000.0) / 500.0
        norm_file_size = torch.tensor(norm_file_size, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, norm_file_size, label
        else:
            return image, norm_file_size


class Mixup:
    """
    Applies Mixup regularization to a batch of data.
    """

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, batch):
        """
        Args:
            batch: tuple of (images, file_sizes, targets)
        Returns:
            images: mixed images
            file_sizes: mixed file sizes
            targets_a: targets for first image
            targets_b: targets for second image
            lam: mixing coefficient
        """
        images, file_sizes, targets = batch

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index, :]
        mixed_file_sizes = lam * file_sizes + (1 - lam) * file_sizes[index]

        targets_a, targets_b = targets, targets[index]

        return mixed_images, mixed_file_sizes, targets_a, targets_b, lam


def get_dataloaders(fold_idx, load_cached_data=True):
    """
    Creates train and validation dataloaders for a specific fold.

    Args:
        fold_idx (int): Index of the fold (0 to N_FOLDS-1).
        load_cached_data (bool): Use caching.

    Returns:
        train_loader, val_loader
    """
    imgs, labels, file_sizes = get_data(mode="train", load_cached_data=load_cached_data)

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(imgs, labels))

    train_idx, val_idx = splits[fold_idx]

    # Split data
    train_imgs, val_imgs = imgs[train_idx], imgs[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]
    train_fs, val_fs = file_sizes[train_idx], file_sizes[val_idx]

    # Create Datasets
    train_ds = CactusDataset(
        train_imgs, train_fs, train_labels, transforms=get_transforms("train")
    )

    val_ds = CactusDataset(
        val_imgs, val_fs, val_labels, transforms=get_transforms("val")
    )

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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
    Creates test dataloader.

    Returns:
        test_loader, test_ids
    """
    imgs, ids, file_sizes = get_data(mode="test", load_cached_data=load_cached_data)

    test_ds = CactusDataset(
        imgs, file_sizes, labels=None, transforms=get_transforms("test")
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, ids
