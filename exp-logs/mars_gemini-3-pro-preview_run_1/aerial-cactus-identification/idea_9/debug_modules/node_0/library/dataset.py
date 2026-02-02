import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config

# Dataset-specific normalization constants derived from analysis
# RGB Mean and Std normalized to [0, 1]
MEAN = [0.50339, 0.45196, 0.46825]
STD = [0.15138, 0.13993, 0.15354]


class CactusDataset(Dataset):
    """
    PyTorch Dataset for Cactus Classification.
    Reads from in-memory numpy arrays to maximize throughput.
    """

    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C), float32 [0, 1].
            labels (np.ndarray, optional): Array of labels with shape (N,).
            transform (A.Compose, optional): Albumentations transform pipeline.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image from memory
        img = self.images[idx]

        # Apply augmentations
        if self.transform:
            # Albumentations expects the image key
            augmented = self.transform(image=img)
            img = augmented["image"]

        # Return image and label (if available)
        if self.labels is not None:
            # Ensure label is float32 for BCEWithLogitsLoss
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label

        return img


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline for the specified phase.

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
        # Validation and Test phases only apply normalization
        return A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Performs Mixup regularization on a batch of data.

    Args:
        x (torch.Tensor): Input images.
        y (torch.Tensor): Target labels.
        alpha (float): Parameter for Beta distribution.
        device (str): Device to perform computation on.

    Returns:
        mixed_x: Mixed input images.
        y_a: Targets for the first image set.
        y_b: Targets for the second image set.
        lam: Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def load_and_cache_data(load_cached_data=True):
    """
    Loads dataset images and labels. Caches them as .npy files to speed up future runs.

    Logic:
    1. If cache exists and load_cached_data is True, load from .npy files.
    2. Otherwise, read metadata, load images using cv2, normalize to [0,1], and save to .npy.
    3. Combines 'train' and 'val' metadata into a single training set for Cross-Validation.

    Returns:
        tuple: ((train_imgs, train_labels), (test_imgs, test_ids))
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = [
        Config.CACHE_TRAIN_IMGS,
        Config.CACHE_TRAIN_LABELS,
        Config.CACHE_TEST_IMGS,
        Config.CACHE_TEST_IDS,
    ]

    cache_exists = all(os.path.exists(f) for f in cache_files)

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {Config.WORKING_DIR}...")
        try:
            train_imgs = np.load(Config.CACHE_TRAIN_IMGS)
            train_labels = np.load(Config.CACHE_TRAIN_LABELS)
            test_imgs = np.load(Config.CACHE_TEST_IMGS)
            test_ids = np.load(Config.CACHE_TEST_IDS)
        except Exception as e:
            print(f"Error loading cache: {e}. Re-processing data...")
            return load_and_cache_data(load_cached_data=False)
    else:
        print("Processing data from scratch...")

        # Load Metadata
        # We combine train and val metadata to allow for custom Cross-Validation splits
        df_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
        df_train_full = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

        df_test = pd.read_csv(Config.TEST_METADATA_PATH)

        def process_images(df, is_test=False):
            imgs = []
            meta_data = []  # labels for train, ids for test

            # Pre-allocate for speed if possible, but list append is fine for 17k images
            for _, row in df.iterrows():
                # Construct full path
                rel_path = row["file_path"]
                full_path = os.path.join(Config.INPUT_DIR, rel_path)

                # Read image
                img = cv2.imread(full_path)
                if img is None:
                    print(f"Warning: Could not read image {full_path}")
                    continue

                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Normalize to float32 [0, 1]
                img = img.astype(np.float32) / 255.0

                imgs.append(img)

                if is_test:
                    meta_data.append(row["id"])
                else:
                    meta_data.append(row["has_cactus"])

            return np.array(imgs, dtype=np.float32), np.array(meta_data)

        print("Loading Training Images...")
        train_imgs, train_labels = process_images(df_train_full, is_test=False)

        print("Loading Test Images...")
        test_imgs, test_ids = process_images(df_test, is_test=True)

        # Save to cache
        print("Saving to cache...")
        np.save(Config.CACHE_TRAIN_IMGS, train_imgs)
        np.save(Config.CACHE_TRAIN_LABELS, train_labels)
        np.save(Config.CACHE_TEST_IMGS, test_imgs)
        np.save(Config.CACHE_TEST_IDS, test_ids)

    # Handle Debug Mode
    if Config.DEBUG:
        print(f"DEBUG MODE: Truncating data to {Config.DEBUG_SAMPLE_SIZE} samples.")
        limit = min(len(train_imgs), Config.DEBUG_SAMPLE_SIZE)
        train_imgs = train_imgs[:limit]
        train_labels = train_labels[:limit]

        limit_test = min(len(test_imgs), Config.DEBUG_SAMPLE_SIZE)
        test_imgs = test_imgs[:limit_test]
        test_ids = test_ids[:limit_test]

    print(f"Data Loaded. Train: {train_imgs.shape}, Test: {test_imgs.shape}")
    return (train_imgs, train_labels), (test_imgs, test_ids)
