import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from library.config import Config
from library.utils import seed_everything


def load_data_to_memory(phase: str, load_cached_data: bool = True):
    """
    Loads image data and labels/ids into memory.
    Caches the processed arrays as .npy files to speed up future runs.

    Args:
        phase (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, targets)
            - images: np.ndarray of shape (N, 3, 32, 32), float32, [0, 1]
            - targets: np.ndarray of labels (int) for train/val, or ids (str) for test.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_imgs_path = os.path.join(Config.CACHE_DIR, f"cache_{phase}_imgs.npy")
    cache_targets_path = os.path.join(Config.CACHE_DIR, f"cache_{phase}_targets.npy")

    # Attempt to load from cache
    if load_cached_data:
        if os.path.exists(cache_imgs_path) and os.path.exists(cache_targets_path):
            print(f"Loading {phase} data from cache...")
            imgs = np.load(cache_imgs_path)
            targets = np.load(
                cache_targets_path, allow_pickle=True
            )  # allow_pickle=True needed for string IDs
            return imgs, targets

    print(f"Processing {phase} data from scratch...")

    # Select metadata file based on phase
    if phase == "train":
        meta_path = Config.TRAIN_META_PATH
    elif phase == "val":
        meta_path = Config.VAL_META_PATH
    elif phase == "test":
        meta_path = Config.TEST_META_PATH
    else:
        raise ValueError(f"Unknown phase: {phase}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)

    # Limit dataset size if debugging
    if Config.DEBUG:
        df = df.head(100)

    img_list = []
    target_list = []

    # Iterate and process
    for _, row in df.iterrows():
        # Construct full path
        # Metadata file_path is relative to input dir (e.g., "train/id.jpg")
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            # Fallback or error handling
            print(f"Warning: Could not read image {full_path}")
            # Create a black image to maintain alignment or skip
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize if necessary (though dataset is 32x32)
        if img.shape[0] != Config.IMG_SIZE or img.shape[1] != Config.IMG_SIZE:
            img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

        # Normalize to [0, 1] and convert to float32
        img = img.astype(np.float32) / 255.0

        # Transpose to Channel-First (C, H, W) for PyTorch
        img = img.transpose(2, 0, 1)

        img_list.append(img)

        # Handle targets
        if phase == "test":
            target_list.append(row["id"])
        else:
            target_list.append(int(row["has_cactus"]))

    # Convert to numpy arrays
    imgs_np = np.array(img_list, dtype=np.float32)

    if phase == "test":
        targets_np = np.array(target_list)  # Array of strings
    else:
        targets_np = np.array(target_list, dtype=np.int64)

    # Save to cache
    np.save(cache_imgs_path, imgs_np)
    np.save(cache_targets_path, targets_np)

    return imgs_np, targets_np


class CactusDataset(Dataset):
    def __init__(self, images, targets, transform=None, is_test=False):
        """
        Args:
            images (np.ndarray): (N, C, H, W) float32 array
            targets (np.ndarray): (N,) int64 array for labels, or string array for IDs
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): If True, returns (img, id). If False, returns (img, label).
        """
        self.images = images
        self.targets = targets
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image tensor (C, H, W)
        img = torch.from_numpy(self.images[idx])

        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            # Return image and ID
            return img, target
        else:
            # Return image and label
            return img, target


def get_transforms(phase: str):
    """
    Returns the transformations for the given phase.
    Since data is pre-loaded as tensors, these transforms operate on tensors.
    """
    if phase == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        # No test-time augmentation in the dataset loader itself;
        # TTA is handled in the inference loop if needed.
        # Data is already normalized and tensor-ready.
        return None


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Returns mixed inputs, pairs of targets, and lambda
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
    Mixup loss function
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_dataloaders(batch_size=None, num_workers=None):
    """
    Factory function to create dataloaders for all phases.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Load data
    train_imgs, train_labels = load_data_to_memory("train")
    val_imgs, val_labels = load_data_to_memory("val")
    test_imgs, test_ids = load_data_to_memory("test")

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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
