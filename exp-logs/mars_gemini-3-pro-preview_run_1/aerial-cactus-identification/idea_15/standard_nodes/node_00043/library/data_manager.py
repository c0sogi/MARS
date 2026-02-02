import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


def load_all_train_data(load_cached_data=True):
    """
    Loads all training and validation data into memory.
    Combines train_metadata.csv and val_metadata.csv.
    Handles caching to .npy files to eliminate I/O latency on subsequent runs.
    """
    cache_exists = (
        os.path.exists(Config.CACHE_TRAIN_IMGS)
        and os.path.exists(Config.CACHE_TRAIN_LABELS)
        and os.path.exists(Config.CACHE_TRAIN_IDS)
        and os.path.exists(Config.CACHE_TRAIN_FSIZES)
    )

    if load_cached_data and cache_exists:
        print("Loading training data from cache...")
        imgs = np.load(Config.CACHE_TRAIN_IMGS)
        labels = np.load(Config.CACHE_TRAIN_LABELS)
        ids = np.load(Config.CACHE_TRAIN_IDS)
        fsizes = np.load(Config.CACHE_TRAIN_FSIZES)
    else:
        print("Processing training data from scratch...")
        # Load metadata
        df_train = pd.read_csv(Config.TRAIN_META_PATH)
        df_val = pd.read_csv(Config.VAL_META_PATH)
        # Combine to form the full training set for Cross-Validation
        df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

        imgs = []
        labels = []
        ids = []
        fsizes = []

        for idx, row in df_full.iterrows():
            img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            # Read Image
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get File Size (in bytes)
            fsize = os.path.getsize(img_path)

            imgs.append(img)
            labels.append(row["has_cactus"])
            ids.append(row["id"])
            fsizes.append(fsize)

        # Convert to numpy arrays
        # Keep images as uint8 to save memory until __getitem__
        imgs = np.array(imgs, dtype=np.uint8)
        labels = np.array(labels, dtype=np.float32)
        ids = np.array(ids)
        fsizes = np.array(fsizes, dtype=np.float32)

        # Save to cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(Config.CACHE_TRAIN_IMGS, imgs)
        np.save(Config.CACHE_TRAIN_LABELS, labels)
        np.save(Config.CACHE_TRAIN_IDS, ids)
        np.save(Config.CACHE_TRAIN_FSIZES, fsizes)
        print("Training data saved to cache.")

    if Config.DEBUG:
        print("DEBUG Mode: Slicing training data to small subset.")
        subset_size = 256
        imgs = imgs[:subset_size]
        labels = labels[:subset_size]
        ids = ids[:subset_size]
        fsizes = fsizes[:subset_size]

    return imgs, labels, ids, fsizes


def load_test_data(load_cached_data=True):
    """
    Loads test data into memory.
    Handles caching to .npy files.
    """
    cache_exists = (
        os.path.exists(Config.CACHE_TEST_IMGS)
        and os.path.exists(Config.CACHE_TEST_IDS)
        and os.path.exists(Config.CACHE_TEST_FSIZES)
    )

    if load_cached_data and cache_exists:
        print("Loading test data from cache...")
        imgs = np.load(Config.CACHE_TEST_IMGS)
        ids = np.load(Config.CACHE_TEST_IDS)
        fsizes = np.load(Config.CACHE_TEST_FSIZES)
        # Create dummy labels for consistency
        labels = np.full(len(imgs), 0.5, dtype=np.float32)
    else:
        print("Processing test data from scratch...")
        df_test = pd.read_csv(Config.TEST_META_PATH)

        imgs = []
        ids = []
        fsizes = []

        for idx, row in df_test.iterrows():
            img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            fsize = os.path.getsize(img_path)

            imgs.append(img)
            ids.append(row["id"])
            fsizes.append(fsize)

        imgs = np.array(imgs, dtype=np.uint8)
        ids = np.array(ids)
        fsizes = np.array(fsizes, dtype=np.float32)
        labels = np.full(len(imgs), 0.5, dtype=np.float32)

        # Save to cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(Config.CACHE_TEST_IMGS, imgs)
        np.save(Config.CACHE_TEST_IDS, ids)
        np.save(Config.CACHE_TEST_FSIZES, fsizes)
        print("Test data saved to cache.")

    if Config.DEBUG:
        print("DEBUG Mode: Slicing test data.")
        subset_size = 100
        imgs = imgs[:subset_size]
        labels = labels[:subset_size]
        ids = ids[:subset_size]
        fsizes = fsizes[:subset_size]

    return imgs, labels, ids, fsizes


def get_file_size_stats(file_sizes):
    """
    Calculates mean and std of file sizes for Z-score normalization.
    Should be calculated on the training set and applied to all sets.
    """
    mean = np.mean(file_sizes)
    std = np.std(file_sizes)
    return mean, std


def normalize_file_sizes(file_sizes, mean, std):
    """
    Applies Z-score normalization to file sizes.
    """
    return (file_sizes - mean) / (std + 1e-8)


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the dataset.

    Args:
        mode: 'train' or 'val'/'test'.
    """
    if mode == "train":
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
    def __init__(self, images, labels, file_sizes, ids, transform=None):
        """
        Args:
            images: numpy array of images (N, H, W, 3) in uint8.
            labels: numpy array of labels (N,).
            file_sizes: numpy array of normalized file sizes (N,).
            ids: numpy array of image IDs (N,).
            transform: Albumentations transform pipeline.
        """
        self.images = images
        self.labels = labels
        self.file_sizes = file_sizes
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        fsize = self.file_sizes[idx]
        img_id = self.ids[idx]

        if self.transform:
            # Albumentations expects HWC image
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback manual conversion if no transform is provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return (
            image,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(fsize, dtype=torch.float32),
            img_id,
        )


def mixup_data(x, y, s, alpha=0.2, device=None):
    """
    Performs Mixup on images and file size features.

    Args:
        x: Input images batch (Batch, C, H, W)
        y: Target labels batch (Batch,)
        s: File size features batch (Batch,)
        alpha: Mixup beta distribution parameter
        device: Torch device

    Returns:
        mixed_x: Mixed images
        mixed_y: Soft labels (linear combination of targets)
        mixed_s: Mixed file size features
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    if device is None:
        device = x.device

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    # Mix images
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Mix labels (soft targets)
    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b

    # Mix file size metadata (conditioning signal)
    mixed_s = lam * s + (1 - lam) * s[index]

    return mixed_x, mixed_y, mixed_s
