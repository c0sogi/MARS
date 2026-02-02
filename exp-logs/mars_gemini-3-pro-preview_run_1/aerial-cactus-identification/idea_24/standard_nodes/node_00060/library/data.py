import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from library.config import Config
from library.utils import set_seed


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Returns mixed inputs, pairs of targets, and lambda.
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


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def load_and_process_data(
    metadata_path, cache_prefix, load_cached_data=True, is_test=False, debug=False
):
    """
    Loads data from disk, processes it into tensors, and caches it to .npy files.

    Args:
        metadata_path: Path to the metadata CSV.
        cache_prefix: Prefix for the cache filenames (e.g., 'train', 'val', 'test').
        load_cached_data: Whether to attempt loading from cache.
        is_test: Whether this is the test set (affects return values).
        debug: If True, only loads a small subset.

    Returns:
        images (np.ndarray): Array of shape (N, 3, 32, 32) float32.
        targets (np.ndarray): Array of labels or IDs.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    imgs_cache_path = os.path.join(cache_dir, f"{cache_prefix}_imgs.npy")
    targets_cache_path = os.path.join(
        cache_dir, f"{cache_prefix}_{'ids' if is_test else 'labels'}.npy"
    )

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(imgs_cache_path)
        and os.path.exists(targets_cache_path)
    ):
        try:
            # print(f"Loading {cache_prefix} data from cache...")
            images = np.load(imgs_cache_path)
            targets = np.load(targets_cache_path)

            if debug:
                return (
                    images[: Config.DEBUG_SUBSET_SIZE],
                    targets[: Config.DEBUG_SUBSET_SIZE],
                )
            return images, targets
        except Exception as e:
            print(f"Failed to load cache for {cache_prefix}: {e}. Recomputing...")

    # 2. Process from scratch
    # print(f"Processing {cache_prefix} data from raw files...")
    df = pd.read_csv(metadata_path)

    img_list = []
    target_list = []

    # Pre-calculate full paths
    # Metadata file_path is relative to input dir (e.g., 'train/id.jpg')
    full_paths = (
        df["file_path"].apply(lambda x: os.path.join(Config.INPUT_DIR, x)).tolist()
    )

    if is_test:
        raw_targets = df["id"].tolist()
    else:
        raw_targets = df["has_cactus"].values.astype(np.float32)

    count = 0
    for idx, path in enumerate(full_paths):
        if not os.path.exists(path):
            continue

        # Read image in BGR
        img = cv2.imread(path)
        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Ensure 32x32 (though dataset is uniform, safety check)
        if img.shape[0] != 32 or img.shape[1] != 32:
            img = cv2.resize(img, (32, 32))

        # Normalize to [0, 1] and CHW format
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        img_list.append(img)
        target_list.append(raw_targets[idx])

        count += 1
        # If not caching (e.g. debug run without cache), we could stop early,
        # but to create a valid cache we must process all unless debug is specifically handled differently.
        # Here we process all to save a valid cache file.

    images = np.array(img_list, dtype=np.float32)
    targets = np.array(target_list)

    # Save to cache
    np.save(imgs_cache_path, images)
    np.save(targets_cache_path, targets)

    if debug:
        return images[: Config.DEBUG_SUBSET_SIZE], targets[: Config.DEBUG_SUBSET_SIZE]

    return images, targets


class CactusDataset(Dataset):
    def __init__(self, images, targets, transform=None, is_test=False):
        """
        Args:
            images: np.ndarray of shape (N, 3, 32, 32)
            targets: np.ndarray of labels (N,) or IDs (N,)
            transform: torchvision transforms
            is_test: bool
        """
        self.images = images
        self.targets = targets
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert numpy array to tensor
        # images are already float32 (C, H, W)
        img = torch.from_numpy(self.images[idx])

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            # Return image and ID for submission
            return img, self.targets[idx]
        else:
            # Return image and label
            label = torch.tensor(self.targets[idx], dtype=torch.float32)
            return img, label


def get_transforms(split="train"):
    """
    Returns the augmentation pipeline.
    Since images are already tensors (C, H, W) in [0, 1], we use tensor-compatible transforms.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # Rotational invariance is high for aerial photos
            ]
        )
    else:
        # No test-time augmentation in the dataset class itself (handled in inference loop if TTA is used)
        # No normalization needed as data is already [0, 1]
        return None


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug: If True, loads a small subset of data.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Data
    train_imgs, train_labels = load_and_process_data(
        Config.TRAIN_METADATA_PATH,
        "train",
        load_cached_data=Config.USE_CACHE,
        is_test=False,
        debug=debug,
    )

    val_imgs, val_labels = load_and_process_data(
        Config.VAL_METADATA_PATH,
        "val",
        load_cached_data=Config.USE_CACHE,
        is_test=False,
        debug=debug,
    )

    test_imgs, test_ids = load_and_process_data(
        Config.TEST_METADATA_PATH,
        "test",
        load_cached_data=Config.USE_CACHE,
        is_test=True,
        debug=debug,
    )

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_labels, transform=get_transforms("train"), is_test=False
    )

    val_dataset = CactusDataset(
        val_imgs, val_labels, transform=get_transforms("val"), is_test=False
    )

    test_dataset = CactusDataset(
        test_imgs, test_ids, transform=get_transforms("test"), is_test=True
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
