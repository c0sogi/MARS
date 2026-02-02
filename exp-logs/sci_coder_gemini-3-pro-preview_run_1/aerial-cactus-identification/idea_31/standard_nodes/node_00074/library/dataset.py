import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms

from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    MIXUP_ALPHA,
)

# Calculated from data analysis
# Mean: R=0.503, G=0.452, B=0.468
# Std : R=0.151, G=0.140, B=0.153
NORM_MEAN = [0.503, 0.452, 0.468]
NORM_STD = [0.151, 0.140, 0.153]


class CactusDataset(Dataset):
    """
    In-memory dataset for Cactus identification.
    Stores images as float tensors to maximize throughput.
    """

    def __init__(self, ids, images, labels=None, transform=None):
        """
        Args:
            ids (np.array): Array of image filenames.
            images (torch.Tensor): Tensor of shape (N, 3, 32, 32).
            labels (torch.Tensor, optional): Tensor of shape (N,).
            transform (callable, optional): Transform to apply to the image.
        """
        self.ids = ids
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Images are already (3, H, W) float tensors in [0, 1]
        img = self.images[idx]

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        else:
            return img, self.ids[idx]


def get_transforms(mode="train"):
    """
    Returns the transformations for the given mode.
    Since images are already tensors in [0, 1], we use torchvision transforms.
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ]
        )
    else:
        return transforms.Compose([transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)])


def mixup_data(x, y, alpha=MIXUP_ALPHA, device="cuda"):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def _load_and_process_data(metadata_paths, cache_prefix, load_cached_data=True):
    """
    Internal function to load data from metadata CSVs, process images, and cache them.

    Args:
        metadata_paths (list): List of paths to metadata CSV files.
        cache_prefix (str): Prefix for cache files (e.g., 'train_full', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        ids (np.array), images (torch.Tensor), labels (torch.Tensor or None)
    """
    # Define cache paths
    cache_ids_path = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")
    cache_imgs_path = os.path.join(CACHE_DIR, f"{cache_prefix}_imgs.npy")
    cache_lbls_path = os.path.join(CACHE_DIR, f"{cache_prefix}_labels.npy")

    # Try loading from cache
    if load_cached_data:
        if os.path.exists(cache_ids_path) and os.path.exists(cache_imgs_path):
            # Check labels existence only if we expect them (not test)
            if "test" in cache_prefix or os.path.exists(cache_lbls_path):
                print(f"Loading {cache_prefix} data from cache...")
                ids = np.load(cache_ids_path, allow_pickle=True)
                images = torch.from_numpy(np.load(cache_imgs_path))

                labels = None
                if os.path.exists(cache_lbls_path):
                    labels = torch.from_numpy(np.load(cache_lbls_path))

                return ids, images, labels

    # If not cached or load failed, process from scratch
    print(f"Processing {cache_prefix} data from scratch...")

    # Combine metadata
    dfs = []
    for path in metadata_paths:
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)

    ids = []
    images_list = []
    labels_list = []

    # Check if labels exist
    has_labels = "has_cactus" in df.columns and "test" not in cache_prefix

    for _, row in df.iterrows():
        img_id = row["id"]
        # Metadata file_path is relative to input dir (e.g., "train/xxx.jpg")
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and convert to float32
        img = img.astype(np.float32) / 255.0

        # Transpose to (C, H, W)
        img = np.transpose(img, (2, 0, 1))

        ids.append(img_id)
        images_list.append(img)

        if has_labels:
            labels_list.append(row["has_cactus"])

    # Convert to numpy/torch
    ids = np.array(ids)
    images = np.stack(images_list)  # (N, 3, 32, 32)

    # Save to cache
    np.save(cache_ids_path, ids)
    np.save(cache_imgs_path, images)

    images = torch.from_numpy(images)

    labels = None
    if has_labels:
        labels = np.array(labels_list, dtype=np.float32)
        np.save(cache_lbls_path, labels)
        labels = torch.from_numpy(labels)

    return ids, images, labels


def get_loaders(
    fold_idx,
    n_folds=5,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    load_cached_data=True,
):
    """
    Creates Stratified K-Fold DataLoaders for training and validation.
    Combines train and val metadata to form the full training set before splitting.
    """
    # Load full training data (Train + Val metadata)
    ids, images, labels = _load_and_process_data(
        [TRAIN_META_PATH, VAL_META_PATH],
        "full_train",
        load_cached_data=load_cached_data,
    )

    if labels is None:
        raise ValueError("Training data must have labels.")

    # Stratified K-Fold Split
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # Get indices for the requested fold
    # labels are float tensors, convert to numpy int for stratification
    y_indices = labels.numpy().astype(int)

    folds = list(skf.split(np.zeros(len(labels)), y_indices))
    train_idx, val_idx = folds[fold_idx]

    # Create subsets
    train_ids, val_ids = ids[train_idx], ids[val_idx]
    train_imgs, val_imgs = images[train_idx], images[val_idx]
    train_lbls, val_lbls = labels[train_idx], labels[val_idx]

    # Create Datasets
    train_dataset = CactusDataset(
        train_ids, train_imgs, train_lbls, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(
        val_ids, val_imgs, val_lbls, transform=get_transforms("val")
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoader for the test set.
    """
    ids, images, _ = _load_and_process_data(
        [TEST_META_PATH], "test", load_cached_data=load_cached_data
    )

    dataset = CactusDataset(ids, images, labels=None, transform=get_transforms("test"))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
