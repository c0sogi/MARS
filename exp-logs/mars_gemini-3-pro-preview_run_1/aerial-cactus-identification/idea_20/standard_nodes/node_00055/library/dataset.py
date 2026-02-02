import os
import numpy as np
from library.utils import load_and_cache_data as utils_load_data
from library.utils import CactusDataset

# Expose CactusDataset for import from this module
__all__ = ["CactusDataset", "load_and_cache_images", "get_transforms"]


def load_and_cache_images(
    input_dir="./input",
    metadata_dir="./metadata",
    cache_dir="./working/idea_20",
    load_cached_data=True,
    max_samples=None,
):
    """
    Loads images and labels, caching them to disk as .npy files to avoid re-reading small files.
    Wraps the library.utils.load_and_cache_data function.

    Args:
        input_dir (str): Path to the input directory containing images.
        metadata_dir (str): Path to the directory containing metadata CSVs.
        cache_dir (str): Directory where .npy cache files will be stored.
        load_cached_data (bool): If True, attempts to load from cache first.
                                 If False or cache missing, re-processes raw images.
        max_samples (int, optional): If provided, limits the number of samples loaded
                                     (for debugging purposes).

    Returns:
        tuple: (train_imgs, train_labels, test_imgs, test_ids)
               train_imgs: np.ndarray of shape (N, 32, 32, 3)
               train_labels: np.ndarray of shape (N,)
               test_imgs: np.ndarray of shape (M, 32, 32, 3)
               test_ids: np.ndarray of shape (M,)
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Call the utility function to handle loading/caching logic
    train_imgs, train_labels, test_imgs, test_ids = utils_load_data(
        input_dir=input_dir,
        metadata_dir=metadata_dir,
        cache_dir=cache_dir,
        load_cached_data=load_cached_data,
    )

    # Apply subsetting if max_samples is specified (for debugging)
    if max_samples is not None and max_samples > 0:
        train_imgs = train_imgs[:max_samples]
        train_labels = train_labels[:max_samples]
        test_imgs = test_imgs[:max_samples]
        test_ids = test_ids[:max_samples]

    return train_imgs, train_labels, test_imgs, test_ids


def get_transforms(split="train"):
    """
    Returns the transform configuration expected by the CactusDataset.

    The provided CactusDataset implementation uses a boolean flag to trigger
    internal hardcoded geometric augmentations (RandomHorizontalFlip, RandomVerticalFlip).

    Args:
        split (str): The data split ('train', 'val', or 'test').

    Returns:
        bool: True if augmentations should be enabled (for training), False otherwise.
    """
    if split == "train":
        return True
    return False
