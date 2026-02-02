import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import configuration
from library.config import PATCH_SIZE

# Import pre-implemented classes and functions from the provided library
# to strictly follow "Import ... instead of re-implementing"
from library.model import DenoisingDataset, get_data


def get_transforms(mode="train"):
    """
    Constructs the Albumentations transform pipeline.

    Args:
        mode (str): 'train' for augmentation, 'val' or 'test' for just normalization/tensor conversion.

    Returns:
        A.Compose: The composition of transforms.
    """
    if mode == "train":
        # Aggressive geometric augmentations as specified in the task
        return A.Compose(
            [
                A.RandomCrop(height=PATCH_SIZE, width=PATCH_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        # No geometric augmentation for validation/test, just tensor conversion
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )


def load_dataset_with_transforms(load_cached_data=True):
    """
    Loads the raw data and wraps it into DenoisingDataset objects with defined transforms.

    This function utilizes the caching mechanism provided by `get_data` in library.model.

    Args:
        load_cached_data (bool): If True, attempts to load from .npz cache.

    Returns:
        tuple: (train_dataset, test_dataset)
            train_dataset: DenoisingDataset with training augmentations.
            test_dataset: DenoisingDataset with test transforms (no augmentation).
    """
    # Load raw numpy arrays using the library function (handles caching and file loading)
    (train_ids, train_noisy, train_clean), (test_ids, test_noisy) = get_data(
        load_cached_data=load_cached_data
    )

    # Get transforms
    train_transform = get_transforms(mode="train")
    test_transform = get_transforms(mode="test")

    # Instantiate Datasets
    # train_dataset includes the clean targets and uses training augmentations
    train_dataset = DenoisingDataset(
        noisy_imgs=train_noisy, clean_imgs=train_clean, transform=train_transform
    )

    # test_dataset has no targets and uses deterministic transforms
    test_dataset = DenoisingDataset(
        noisy_imgs=test_noisy, clean_imgs=None, transform=test_transform
    )

    return train_dataset, test_dataset
