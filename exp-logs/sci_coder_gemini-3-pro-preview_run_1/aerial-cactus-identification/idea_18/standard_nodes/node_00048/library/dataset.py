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

# Initialize logger
logger = get_logger("dataset")


def get_transforms(mode="train", in_chans=3):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
        in_chans (int): 3 for RGB, 4 for RGB + Laplacian.
    """
    # Base Normalization stats from Config
    mean = list(Config.NORM_MEAN)
    std = list(Config.NORM_STD)

    # If 4 channels, append stats for the Laplacian channel.
    # Laplacian is computed on grayscale, normalized to roughly 0-1 range.
    # We use 0.5 mean/std as a safe default for this structural feature.
    if in_chans == 4:
        mean.append(0.5)
        std.append(0.5)

    transforms_list = []

    if mode == "train":
        # Geometric augmentations for training
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        )

    # Normalization and Tensor conversion apply to all modes
    transforms_list.extend(
        [A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0), ToTensorV2()]
    )

    return A.Compose(transforms_list)


def compute_laplacian(image_rgb):
    """
    Computes the Laplacian edge map for an RGB image.

    Args:
        image_rgb (np.ndarray): Image in RGB format (H, W, 3).

    Returns:
        np.ndarray: Laplacian map (H, W, 1) normalized to 0-255 range for consistency.
    """
    # Convert to Grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Compute Laplacian (ksize=3 is standard for 3x3 kernels)
    # CV_64F to avoid overflow/underflow during calculation
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

    # Take absolute value to capture edge magnitude
    laplacian = np.absolute(laplacian)

    # Normalize to 0-255 uint8 range to match RGB channels
    # We clip to ensure bounds, though typical laplacian values are small
    laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)

    return laplacian[..., np.newaxis]


def load_data_to_memory(metadata_path, cache_keys, load_cached_data=True):
    """
    Loads images and metadata into RAM, using caching to speed up subsequent runs.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_keys (dict): Dictionary containing keys for cache paths
                           (e.g., 'imgs', 'labels', 'fsizes', 'ids').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, file_sizes, ids)
               images: np.ndarray of shape (N, H, W, 3)
               labels: np.ndarray of shape (N,)
               file_sizes: np.ndarray of shape (N,)
               ids: np.ndarray of shape (N,)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Paths for cache files
    path_imgs = Config.get_cache_path(cache_keys.get("imgs"))
    path_labels = Config.get_cache_path(cache_keys.get("labels"))
    path_fsizes = Config.get_cache_path(cache_keys.get("fsizes"))
    path_ids = Config.get_cache_path(cache_keys.get("ids"))

    # Check if all cache files exist
    # Cite debug_lesson_17: Prevent vacuous truth (all([]) is True) when all paths are None
    paths_to_check = [
        p for p in [path_imgs, path_labels, path_fsizes, path_ids] if p is not None
    ]
    cache_exists = len(paths_to_check) > 0 and all(
        os.path.exists(p) for p in paths_to_check
    )

    if load_cached_data and cache_exists:
        logger.info(f"Loading cached data from {Config.CACHE_DIR}...")
        images = np.load(path_imgs)
        labels = np.load(path_labels) if path_labels else None
        file_sizes = np.load(path_fsizes)
        ids = np.load(path_ids)
        return images, labels, file_sizes, ids

    logger.info(f"Processing data from {metadata_path}...")

    # Load Metadata
    df = pd.read_csv(metadata_path)

    img_list = []
    label_list = []
    fsize_list = []
    id_list = []

    # Iterate through metadata
    for _, row in df.iterrows():
        # Construct full path
        # Metadata file_path is relative to input dir (e.g., "train/xxx.jpg")
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read Image
        img = cv2.imread(full_path)
        if img is None:
            logger.warning(f"Could not read image: {full_path}")
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get File Size
        fsize = os.path.getsize(full_path)

        img_list.append(img)
        fsize_list.append(fsize)
        id_list.append(row["id"])

        # Handle Labels (Test set might have placeholder)
        if "has_cactus" in row:
            label_list.append(row["has_cactus"])
        else:
            label_list.append(-1)  # Placeholder

    # Convert to Numpy Arrays
    images = np.array(img_list, dtype=np.uint8)
    file_sizes = np.array(fsize_list, dtype=np.float32)
    ids = np.array(id_list)
    labels = np.array(label_list, dtype=np.float32) if label_list else None

    # Save to Cache
    logger.info(f"Saving data to cache at {Config.CACHE_DIR}...")
    if path_imgs:
        np.save(path_imgs, images)
    if path_fsizes:
        np.save(path_fsizes, file_sizes)
    if path_ids:
        np.save(path_ids, ids)
    if path_labels and labels is not None:
        np.save(path_labels, labels)

    return images, labels, file_sizes, ids


class CactusDataset(Dataset):
    """
    PyTorch Dataset for Cactus Identification.
    Supports Dual-Domain input (Spatial RGB or Texture RGB+Laplacian).
    """

    def __init__(
        self, images, labels, file_sizes, ids, transform=None, in_chans=3, fs_stats=None
    ):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, 3).
            labels (np.ndarray): Array of labels (N,).
            file_sizes (np.ndarray): Array of file sizes in bytes (N,).
            ids (np.ndarray): Array of image IDs (N,).
            transform (A.Compose): Albumentations transforms.
            in_chans (int): 3 for RGB, 4 for RGB + Laplacian.
            fs_stats (tuple): (mean, std) for file size normalization (Z-score).
                              If None, computed from this dataset (avoid for validation/test).
        """
        self.images = images
        self.labels = labels
        self.file_sizes = file_sizes
        self.ids = ids
        self.transform = transform
        self.in_chans = in_chans

        # File Size Statistics for Normalization (FiLM)
        if fs_stats is None:
            self.fs_mean = np.mean(file_sizes)
            self.fs_std = np.std(file_sizes) + 1e-6
        else:
            self.fs_mean, self.fs_std = fs_stats

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]  # (H, W, 3) RGB
        fsize_bytes = self.file_sizes[idx]

        # 1. Texture Domain Processing
        if self.in_chans == 4:
            # Compute Laplacian Edge Map
            laplacian = compute_laplacian(image)  # (H, W, 1)
            # Concatenate to make 4-channel input
            image = np.concatenate([image, laplacian], axis=-1)

        # 2. Augmentations & Normalization
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # 3. Meta-Features Processing

        # A. For FiLM: Z-score normalized file size
        fsize_norm = (fsize_bytes - self.fs_mean) / self.fs_std

        # B. For MTL (Auxiliary Loss): Log-transformed file size
        # We use log1p to handle potential zeros and compress range
        # Scaled roughly to be in a similar range to BCE loss (0-1ish)
        # Max file size is ~10kB, log(10000) ~ 9.2. Dividing by 10 puts it in [0, 1]
        fsize_log = np.log1p(fsize_bytes) / 10.0

        # Prepare output dictionary
        sample = {
            "image": image,
            "file_size_norm": torch.tensor(fsize_norm, dtype=torch.float32),
            "file_size_log": torch.tensor(fsize_log, dtype=torch.float32),
            "id": self.ids[idx],
        }

        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return sample
