import os
import torch
import numpy as np
import pandas as pd
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_TRAIN_IMAGES,
    CACHE_TRAIN_IDS,
    CACHE_TRAIN_LABELS,
    CACHE_VAL_IMAGES,
    CACHE_VAL_IDS,
    CACHE_VAL_LABELS,
    CACHE_TEST_IMAGES,
    CACHE_TEST_IDS,
    BATCH_SIZE,
    NUM_WORKERS,
    INPUT_DROPOUT_PROB,
    IMG_SIZE,
    SEED,
    DEBUG,
)
from library.data_processing import process_dataset


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Rotate limit 30 degrees. border_mode=0 is cv2.BORDER_CONSTANT (black padding)
                A.Rotate(limit=30, p=0.5, border_mode=0, value=0),
                # Elastic Transform: Alpha/Sigma tuned for MRI deformations
                A.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=50, p=0.5, border_mode=0, value=0
                ),
                # Grid Distortion
                A.GridDistortion(
                    num_steps=5, distort_limit=0.3, p=0.5, border_mode=0, value=0
                ),
                # Strictly Exclude: RandomScale, ShiftScaleRotate (with shift/scale), Translate
                # Convert to Tensor (HWC -> CHW)
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for the ROI-Normalized Volumetric Stack.
    Handles 9-channel input tensors with Structured Input Dropout.
    """

    def __init__(
        self, images, labels=None, ids=None, transform=None, input_dropout_prob=0.0
    ):
        """
        Args:
            images (np.ndarray): Array of shape (N, H, W, 9).
            labels (np.ndarray, optional): Array of shape (N,).
            ids (np.ndarray, optional): Array of BraTS21IDs.
            transform (A.Compose): Albumentations transforms.
            input_dropout_prob (float): Probability of applying structured input dropout.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.input_dropout_prob = input_dropout_prob

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image: (H, W, 9)
        img = self.images[idx]

        # Apply Albumentations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]  # Returns Tensor (9, H, W) due to ToTensorV2

        # Apply Structured Input Dropout (Regularization)
        # We assume img is now a Tensor (C, H, W)
        if self.input_dropout_prob > 0:
            if torch.rand(1).item() < self.input_dropout_prob:
                # Randomly choose between Center or Periphery
                if torch.rand(1).item() < 0.5:
                    # Drop Center Triplet (Channels 3, 4, 5)
                    # Corresponds to Depth 50% [FLAIR, T1wCE, T2w]
                    img[3:6, :, :] = 0.0
                else:
                    # Drop Peripheral Triplets (Channels 0-2 and 6-8)
                    # Corresponds to Depth 40% and 60%
                    img[0:3, :, :] = 0.0
                    img[6:9, :, :] = 0.0

        # Handle Labels vs Inference Mode
        if self.labels is not None:
            label = self.labels[idx]
            # Return (Image, Label)
            # Label needs to be float tensor for BCEWithLogits
            return img, torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        else:
            # Return (Image, ID) for submission mapping
            subject_id = self.ids[idx] if self.ids is not None else idx
            return img, subject_id


def get_dataloaders(
    debug=DEBUG, load_cached=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
):
    """
    Orchestrates data loading, processing, and DataLoader creation.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load Metadata
    df_train = pd.read_csv(TRAIN_METADATA_PATH)
    df_val = pd.read_csv(VAL_METADATA_PATH)
    df_test = pd.read_csv(TEST_METADATA_PATH)

    # 2. Process Datasets (Load from cache or compute)
    # Train
    train_ids, train_images, train_labels = process_dataset(
        df_train,
        CACHE_TRAIN_IDS,
        CACHE_TRAIN_IMAGES,
        CACHE_TRAIN_LABELS,
        load_cached_data=load_cached,
        debug=debug,
    )

    # Validation
    val_ids, val_images, val_labels = process_dataset(
        df_val,
        CACHE_VAL_IDS,
        CACHE_VAL_IMAGES,
        CACHE_VAL_LABELS,
        load_cached_data=load_cached,
        debug=debug,
    )

    # Test
    test_ids, test_images, _ = process_dataset(
        df_test,
        CACHE_TEST_IDS,
        CACHE_TEST_IMAGES,
        cache_labels_path=None,  # No labels for test
        load_cached_data=load_cached,
        debug=debug,
    )

    # 3. Create Dataset Objects
    train_dataset = BraTSDataset(
        images=train_images,
        labels=train_labels,
        ids=train_ids,
        transform=get_transforms("train"),
        input_dropout_prob=INPUT_DROPOUT_PROB,
    )

    val_dataset = BraTSDataset(
        images=val_images,
        labels=val_labels,
        ids=val_ids,
        transform=get_transforms("val"),
        input_dropout_prob=0.0,  # Disable dropout for validation
    )

    test_dataset = BraTSDataset(
        images=test_images,
        labels=None,
        ids=test_ids,
        transform=get_transforms("test"),
        input_dropout_prob=0.0,  # Disable dropout for inference
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to stabilize BatchNorm
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
