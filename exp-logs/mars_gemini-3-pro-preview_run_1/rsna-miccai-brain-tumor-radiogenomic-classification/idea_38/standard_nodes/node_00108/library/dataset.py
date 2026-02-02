import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library.config import (
    IMG_SIZE,
    SEED,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_CACHE_FILE,
    VAL_CACHE_FILE,
    TEST_CACHE_FILE,
    DEBUG,
    DEBUG_DATA_SIZE,
)
from library.dicom_processing import generate_dataset


def get_transforms(phase="train"):
    """
    Returns the Spatially-Constrained Augmentation pipeline.
    Strictly excludes Random Translations (Shift) and Scaling to preserve
    the anatomical anchoring (Center of Mass) established in preprocessing.
    """
    if phase == "train":
        return A.Compose(
            [
                # Spatial transformations that preserve the center anchor
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),  # Small rotation around center
                # Non-rigid deformations
                # Corrected alpha to apply effective deformation (Cite Lesson 00102)
                A.ElasticTransform(alpha=100, sigma=10, alpha_affine=10, p=0.5),
                A.GridDistortion(p=0.5),
                # Convert to Tensor (HWC -> CHW)
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: No augmentation, just tensor conversion
        return A.Compose([ToTensorV2()])


class VAADataset(Dataset):
    """
    Verified Anatomically-Anchored (VAA) Dataset.
    Wraps pre-processed, CoM-anchored 3-channel MRI images.
    """

    def __init__(self, images, labels, ids, transform=None):
        """
        Args:
            images (np.ndarray): Array of shape (N, H, W, 3), float32, [0, 1].
            labels (np.ndarray): Array of shape (N,), float32.
            ids (np.ndarray): Array of shape (N,), int64.
            transform (albumentations.Compose): Augmentation pipeline.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and label
        image = self.images[idx]  # (H, W, 3)
        label = self.labels[idx]
        braTS21ID = self.ids[idx]

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Ensure label is a tensor
        label = torch.tensor(label, dtype=torch.float32)

        # Return dictionary for flexibility
        return {
            "image": image,  # (3, H, W)
            "label": label,  # (1,)
            "BraTS21ID": braTS21ID,
        }


def get_datasets(load_cached_data=True):
    """
    Loads raw data, processes it (or loads from cache), and returns
    VAADataset objects for train, validation, and test sets.
    """
    # 1. Load/Generate Training Data
    print("Preparing Training Dataset...")
    train_imgs, train_lbls, train_ids = generate_dataset(
        TRAIN_METADATA_PATH,
        TRAIN_CACHE_FILE,
        load_cached_data=load_cached_data,
        debug=DEBUG,
        debug_size=DEBUG_DATA_SIZE,
    )

    # 2. Load/Generate Validation Data
    print("Preparing Validation Dataset...")
    val_imgs, val_lbls, val_ids = generate_dataset(
        VAL_METADATA_PATH,
        VAL_CACHE_FILE,
        load_cached_data=load_cached_data,
        debug=DEBUG,
        debug_size=DEBUG_DATA_SIZE,
    )

    # 3. Load/Generate Test Data
    print("Preparing Test Dataset...")
    test_imgs, test_lbls, test_ids = generate_dataset(
        TEST_METADATA_PATH,
        TEST_CACHE_FILE,
        load_cached_data=load_cached_data,
        debug=DEBUG,  # Apply debug limit to test as well if debugging
        debug_size=DEBUG_DATA_SIZE,
    )

    # 4. Create Dataset Objects
    train_dataset = VAADataset(
        train_imgs, train_lbls, train_ids, transform=get_transforms("train")
    )

    val_dataset = VAADataset(
        val_imgs, val_lbls, val_ids, transform=get_transforms("valid")
    )

    test_dataset = VAADataset(
        test_imgs, test_lbls, test_ids, transform=get_transforms("test")
    )

    return train_dataset, val_dataset, test_dataset
