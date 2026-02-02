import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


class CactusDataset(Dataset):
    """
    Dataset class that holds images and metadata in RAM.
    """

    def __init__(self, images, file_sizes, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            file_sizes (np.ndarray): Array of normalized file sizes (N,).
            labels (np.ndarray, optional): Array of labels (N,).
            transform (albumentations.Compose, optional): Augmentation pipeline.
        """
        self.images = images
        self.file_sizes = file_sizes
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image and metadata from RAM
        image = self.images[idx]
        file_size = self.file_sizes[idx]

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Convert file_size to tensor
        file_size = torch.tensor(file_size, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, file_size, label
        else:
            return image, file_size


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [A.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD), ToTensorV2()]
        )


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Applies Mixup augmentation to a batch.
    Returns:
        mixed_x: The mixed images.
        y_a: Labels of the first image set.
        y_b: Labels of the second image set.
        lam: The mixing coefficient.
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


def _load_and_cache_split(
    metadata_path,
    cache_img_path,
    cache_label_path,
    cache_fs_path,
    cache_id_path=None,
    load_cached_data=True,
):
    """
    Internal helper to load data from cache or process from scratch.
    Strictly follows the requirement: Try load -> If fail, Process & Save.
    """
    # 1. Try to load from cache
    if load_cached_data:
        # Check if basic files exist
        files_exist = os.path.exists(cache_img_path) and os.path.exists(cache_fs_path)

        # Check optional files
        if cache_label_path:
            files_exist = files_exist and os.path.exists(cache_label_path)
        if cache_id_path:
            files_exist = files_exist and os.path.exists(cache_id_path)

        if files_exist:
            print(f"Loading cached data from {os.path.dirname(cache_img_path)}...")
            images = np.load(cache_img_path)
            file_sizes = np.load(cache_fs_path)
            labels = np.load(cache_label_path) if cache_label_path else None
            ids = np.load(cache_id_path, allow_pickle=True) if cache_id_path else None
            return images, file_sizes, labels, ids

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    img_list = []
    fs_list = []
    label_list = []
    id_list = []

    for _, row in df.iterrows():
        # Construct full path
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read Image
        img = cv2.imread(full_path)
        if img is None:
            # Handle missing images gracefully (though metadata check passed)
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get File Size
        fsize = os.path.getsize(full_path)

        img_list.append(img)
        fs_list.append(fsize)
        id_list.append(row["id"])

        if "has_cactus" in row:
            label_list.append(row["has_cactus"])

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.uint8)
    file_sizes = np.array(fs_list, dtype=np.float32)
    labels = np.array(label_list, dtype=np.float32) if label_list else None
    ids = np.array(id_list)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_img_path), exist_ok=True)
    np.save(cache_img_path, images)
    np.save(cache_fs_path, file_sizes)
    if labels is not None and cache_label_path:
        np.save(cache_label_path, labels)
    if ids is not None and cache_id_path:
        np.save(cache_id_path, ids)

    return images, file_sizes, labels, ids


def get_train_val_loaders(load_cached_data=True):
    """
    Loads train and validation data, normalizes metadata, and returns DataLoaders.

    Returns:
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        stats: Tuple (mean, std) of file sizes from the training set.
    """
    # Load Train Data
    train_imgs, train_fs, train_labels, _ = _load_and_cache_split(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_IMGS,
        Config.CACHE_TRAIN_LABELS,
        Config.CACHE_TRAIN_FILESIZES,
        load_cached_data=load_cached_data,
    )

    # Load Val Data
    val_imgs, val_fs, val_labels, _ = _load_and_cache_split(
        Config.VAL_METADATA_PATH,
        Config.CACHE_VAL_IMGS,
        Config.CACHE_VAL_LABELS,
        Config.CACHE_VAL_FILESIZES,
        load_cached_data=load_cached_data,
    )

    # Handle Debug Mode (Slice data AFTER loading to preserve cache integrity)
    if Config.DEBUG:
        print(f"Debug mode enabled. Slicing dataset to {Config.DEBUG_SAMPLES} samples.")
        train_imgs = train_imgs[: Config.DEBUG_SAMPLES]
        train_fs = train_fs[: Config.DEBUG_SAMPLES]
        train_labels = train_labels[: Config.DEBUG_SAMPLES]

        val_imgs = val_imgs[: Config.DEBUG_SAMPLES]
        val_fs = val_fs[: Config.DEBUG_SAMPLES]
        val_labels = val_labels[: Config.DEBUG_SAMPLES]

    # Calculate Statistics on Training Set
    fs_mean = np.mean(train_fs)
    fs_std = np.std(train_fs) + 1e-8

    # Normalize File Sizes
    train_fs_norm = (train_fs - fs_mean) / fs_std
    val_fs_norm = (val_fs - fs_mean) / fs_std

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_fs_norm, train_labels, transform=get_transforms("train")
    )

    val_dataset = CactusDataset(
        val_imgs, val_fs_norm, val_labels, transform=get_transforms("val")
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, (fs_mean, fs_std)


def get_test_loader(fs_stats, load_cached_data=True):
    """
    Loads test data, applies provided normalization stats, and returns DataLoader.

    Args:
        fs_stats: Tuple (mean, std) calculated from training set.
        load_cached_data: Whether to use cached files.

    Returns:
        test_loader: DataLoader for testing.
        test_ids: Array of image IDs corresponding to the loader order.
    """
    fs_mean, fs_std = fs_stats

    # Load Test Data
    test_imgs, test_fs, _, test_ids = _load_and_cache_split(
        Config.TEST_METADATA_PATH,
        Config.CACHE_TEST_IMGS,
        None,  # No labels for test
        Config.CACHE_TEST_FILESIZES,
        Config.CACHE_TEST_IDS,
        load_cached_data=load_cached_data,
    )

    # Normalize File Sizes using Train Stats
    test_fs_norm = (test_fs - fs_mean) / fs_std

    # Create Dataset
    test_dataset = CactusDataset(
        test_imgs, test_fs_norm, labels=None, transform=get_transforms("test")
    )

    # Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, test_ids
