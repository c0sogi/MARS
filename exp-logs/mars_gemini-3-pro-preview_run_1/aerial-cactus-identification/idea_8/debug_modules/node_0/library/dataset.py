import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_8"

# Normalization stats from data analysis
# Mean: R=128.37, G=115.25, B=119.40 -> Scaled to [0,1]
# Std: R=38.60, G=35.68, B=39.15 -> Scaled to [0,1]
MEAN = (0.503, 0.452, 0.468)
STD = (0.151, 0.140, 0.154)


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_transforms(phase="train"):
    """
    Returns the transformation pipeline for the specified phase.
    Since data is pre-loaded as float tensors (C, H, W) in [0, 1],
    we only need geometric augmentations and normalization.
    """
    if phase == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    else:
        return transforms.Compose([transforms.Normalize(MEAN, STD)])


def load_and_cache_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata CSV, reads images, and caches them as .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cache files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images_npy, labels_npy, ids_npy)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    img_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_imgs.npy")
    label_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_labels.npy")
    id_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(img_cache_path):
        # print(f"Loading {cache_prefix} data from cache...")
        imgs = np.load(img_cache_path)

        labels = None
        if os.path.exists(label_cache_path):
            labels = np.load(label_cache_path)

        ids = None
        if os.path.exists(id_cache_path):
            ids = np.load(id_cache_path)

        return imgs, labels, ids

    # 2. Process from scratch
    # print(f"Processing {cache_prefix} data from {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    img_list = []
    label_list = []
    id_list = []

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            # print(f"Warning: Could not read image {full_path}")
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)

        id_list.append(row["id"])

        # Handle labels if present
        if "has_cactus" in row:
            label_list.append(row["has_cactus"])

    # Convert to numpy arrays
    # Keep as uint8 (N, H, W, C) to save disk space in cache
    imgs = np.array(img_list, dtype=np.uint8)
    ids = np.array(id_list)

    # Save to cache
    np.save(img_cache_path, imgs)
    np.save(id_cache_path, ids)

    labels = None
    if label_list:
        labels = np.array(label_list, dtype=np.float32)
        np.save(label_cache_path, labels)

    return imgs, labels, ids


class CactusDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C) in uint8.
            labels (np.ndarray): Array of labels (N,).
            transform (callable): Transform to apply.
        """
        # Convert to Float Tensor (N, C, H, W) normalized to [0, 1]
        # This loads the entire dataset into RAM as float tensors as requested.
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0

        self.labels = None
        if labels is not None:
            self.labels = torch.from_numpy(labels).float()

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            return img, self.labels[idx]
        else:
            return img, torch.tensor(-1.0)  # Dummy label


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
