import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import get_logger

logger = get_logger(__name__)


def load_and_cache_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads images and metadata, caching them as .npy files to speed up subsequent runs.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cache files (e.g., 'train', 'val', 'test').
        load_cached_data (str): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, file_sizes, ids)
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    p_imgs = os.path.join(cache_dir, f"{cache_prefix}_imgs.npy")
    p_labels = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")
    p_fsizes = os.path.join(cache_dir, f"{cache_prefix}_fsizes.npy")
    p_ids = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # Attempt to load from cache
    if load_cached_data:
        if all(os.path.exists(p) for p in [p_imgs, p_labels, p_fsizes, p_ids]):
            logger.info(f"Loading {cache_prefix} data from cache...")
            imgs = np.load(p_imgs)
            labels = np.load(p_labels)
            fsizes = np.load(p_fsizes)
            ids = np.load(p_ids)
            return imgs, labels, fsizes, ids
        else:
            logger.info(f"Cache missing for {cache_prefix}, processing from scratch...")
    else:
        logger.info(f"Ignoring cache for {cache_prefix}, processing from scratch...")

    # Load metadata
    df = pd.read_csv(metadata_path)

    img_list = []
    label_list = []
    fsize_list = []
    id_list = []

    # Process each row
    for _, row in df.iterrows():
        img_id = row["id"]
        label = row["has_cactus"]
        rel_path = row["file_path"]

        # Construct full path (input dir + relative path from metadata)
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read Image
        if not os.path.exists(full_path):
            # Fallback or skip - strictly speaking should not happen based on metadata validation
            continue

        # Load BGR, convert to RGB
        img = cv2.imread(full_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Ensure 32x32 (though dataset analysis confirmed they are)
        if img.shape[:2] != (32, 32):
            img = cv2.resize(img, (32, 32))

        # Normalize to 0-1 float32
        img = img.astype(np.float32) / 255.0

        # Get file size in bytes
        fsize = os.path.getsize(full_path)

        img_list.append(img)
        label_list.append(label)
        fsize_list.append(fsize)
        id_list.append(img_id)

    # Convert to numpy arrays
    imgs = np.array(img_list, dtype=np.float32)
    labels = np.array(label_list, dtype=np.float32)  # Float for BCE/Mixup
    fsizes = np.array(fsize_list, dtype=np.float32)
    ids = np.array(id_list)

    # Save to cache
    logger.info(f"Saving {cache_prefix} data to cache at {cache_dir}...")
    np.save(p_imgs, imgs)
    np.save(p_labels, labels)
    np.save(p_fsizes, fsizes)
    np.save(p_ids, ids)

    return imgs, labels, fsizes, ids


def mixup_data(x, y, alpha=0.2, device="cuda"):
    """
    Applies Mixup regularization to the batch.
    Returns:
        mixed_x: Mixed inputs
        y_a: Targets for the first image
        y_b: Targets for the second image
        lam: Mixing coefficient
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


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Images are already 0-1 float32.
                # We can apply normalization if we want to shift mean/std,
                # but simple 0-1 is often sufficient for these custom aerials.
                # We will stick to ToTensorV2 which converts HWC to CHW.
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


class CactusDataset(Dataset):
    def __init__(
        self, images, labels, file_sizes, ids, transform=None, fsize_stats=None
    ):
        """
        Args:
            images (np.array): Shape (N, 32, 32, 3) float32
            labels (np.array): Shape (N,)
            file_sizes (np.array): Shape (N,) raw bytes
            ids (np.array): Shape (N,) string IDs
            transform (albumentations.Compose): Augmentation pipeline
            fsize_stats (dict): Statistics for normalization {'mean':, 'std':, 'log_min':, 'log_max':}
        """
        self.images = images
        self.labels = labels
        self.file_sizes = file_sizes
        self.ids = ids
        self.transform = transform
        self.fsize_stats = fsize_stats if fsize_stats else {}

        # Pre-calculate derived file size features to save time in __getitem__
        # 1. FiLM Input: Z-score normalized
        mean = self.fsize_stats.get("mean", 0.0)
        std = self.fsize_stats.get("std", 1.0)
        # Avoid division by zero
        if std == 0:
            std = 1.0
        self.film_inputs = (self.file_sizes - mean) / std

        # 2. MTL Target: Log-transformed and normalized 0-1
        log_min = self.fsize_stats.get("log_min", 0.0)
        log_max = self.fsize_stats.get("log_max", 1.0)
        log_range = log_max - log_min
        if log_range == 0:
            log_range = 1.0

        self.log_fsizes = np.log1p(self.file_sizes)
        self.mtl_targets = (self.log_fsizes - log_min) / log_range

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]  # HWC, 0-1 float
        label = self.labels[idx]
        film_input = self.film_inputs[idx]
        mtl_target = self.mtl_targets[idx]
        img_id = self.ids[idx]

        # Apply transforms
        if self.transform:
            # Albumentations expects numpy HWC
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Ensure label is float tensor for BCE
        label = torch.tensor(label, dtype=torch.float32)
        film_input = torch.tensor(film_input, dtype=torch.float32)
        mtl_target = torch.tensor(mtl_target, dtype=torch.float32)

        return image, label, film_input, mtl_target, img_id


def get_datasets(load_cached_data=True):
    """
    Main entry point to load data and create Dataset objects.
    Computes statistics on Training set and applies them to Val/Test.
    """
    # 1. Load all data arrays
    train_imgs, train_lbls, train_fsizes, train_ids = load_and_cache_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, val_fsizes, val_ids = load_and_cache_data(
        Config.VAL_METADATA_PATH, "val", load_cached_data
    )
    test_imgs, test_lbls, test_fsizes, test_ids = load_and_cache_data(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    # 2. Compute Statistics on TRAIN set only
    fsize_mean = np.mean(train_fsizes)
    fsize_std = np.std(train_fsizes)

    train_log_fsizes = np.log1p(train_fsizes)
    log_min = np.min(train_log_fsizes)
    log_max = np.max(train_log_fsizes)

    stats = {
        "mean": fsize_mean,
        "std": fsize_std,
        "log_min": log_min,
        "log_max": log_max,
    }

    logger.info(f"File Size Stats (Train): Mean={fsize_mean:.2f}, Std={fsize_std:.2f}")
    logger.info(f"Log File Size Stats (Train): Min={log_min:.4f}, Max={log_max:.4f}")

    # 3. Create Datasets
    train_ds = CactusDataset(
        train_imgs,
        train_lbls,
        train_fsizes,
        train_ids,
        transform=get_transforms("train"),
        fsize_stats=stats,
    )

    val_ds = CactusDataset(
        val_imgs,
        val_lbls,
        val_fsizes,
        val_ids,
        transform=get_transforms("val"),
        fsize_stats=stats,
    )

    test_ds = CactusDataset(
        test_imgs,
        test_lbls,
        test_fsizes,
        test_ids,
        transform=get_transforms("test"),  # No augmentation for test
        fsize_stats=stats,
    )

    return train_ds, val_ds, test_ds
