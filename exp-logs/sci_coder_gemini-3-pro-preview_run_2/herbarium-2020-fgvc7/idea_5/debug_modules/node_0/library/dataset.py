import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import (
    INPUT_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED,
)
from library.taxonomy import TaxonomyManager


class HerbariumDataset(Dataset):
    """
    Dataset class for the Herbarium 2020 competition.
    Loads images and provides hierarchical labels (species, genus, family).
    """

    def __init__(self, csv_path, transform=None, is_test=False):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            transform (albumentations.Compose): Augmentation pipeline.
            is_test (bool): If True, returns image_id instead of labels.
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.is_test = is_test

        # Initialize taxonomy manager for label retrieval if we are in training/validation mode
        if not self.is_test:
            # We assume the taxonomy mapping is already generated/cached by the training script setup
            self.taxonomy = TaxonomyManager(load_cached_data=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # The CSV contains paths relative to the input directory (e.g., nybg2020/train/...)
        image_path = os.path.join(INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(image_path)

        # Handle potential missing images (though verification script ensures existence)
        if image is None:
            # Return a black image of correct size to prevent crashing
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback basic transform
            basic_transform = A.Compose(
                [A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.Normalize(), ToTensorV2()]
            )
            image = basic_transform(image=image)["image"]

        if self.is_test:
            # For test set, return image and the image_id needed for submission
            return image, row["image_id"]
        else:
            # For train/val, retrieve hierarchical labels
            species_id = int(row["category_id"])

            # Retrieve parent taxa IDs using the TaxonomyManager
            genus_id = self.taxonomy.get_genus_id(species_id)
            family_id = self.taxonomy.get_family_id(species_id)

            # Return tuple of labels: (Species, Genus, Family)
            return image, (species_id, genus_id, family_id)


def get_transforms(image_size, mode="train"):
    """
    Returns the albumentations transform pipeline for a given mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test transforms (Deterministic)
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    train_csv=TRAIN_CSV,
    val_csv=VAL_CSV,
    test_csv=TEST_CSV,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
):
    """
    Constructs and returns DataLoader objects for train, validation, and test sets.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define transforms
    train_transform = get_transforms(image_size, mode="train")
    val_transform = get_transforms(image_size, mode="val")

    # Initialize Datasets
    train_dataset = HerbariumDataset(
        csv_path=train_csv, transform=train_transform, is_test=False
    )

    val_dataset = HerbariumDataset(
        csv_path=val_csv, transform=val_transform, is_test=False
    )

    test_dataset = HerbariumDataset(
        csv_path=test_csv, transform=val_transform, is_test=True
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
