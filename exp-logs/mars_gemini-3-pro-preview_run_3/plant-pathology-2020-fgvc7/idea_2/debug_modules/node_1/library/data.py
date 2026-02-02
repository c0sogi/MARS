import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import (
    INPUT_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    IMG_SIZE,
    CLASS_LABELS,
    BATCH_SIZE,
    NUM_WORKERS,
    seed_everything,
    SEED,
)


class AppleDataset(Dataset):
    """
    Custom Dataset for Apple Disease Detection.
    Loads images via OpenCV and applies Albumentations transforms.
    """

    def __init__(self, df, transforms=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_path, labels).
            transforms (albumentations.Compose): Transformations to apply.
            is_test (bool): If True, returns (image, image_id). If False, returns (image, label).
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.is_test = is_test

        # Pre-check columns to ensure safety
        if not self.is_test:
            missing_cols = [c for c in CLASS_LABELS if c not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing label columns in dataframe: {missing_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative (e.g., images/Train_0.jpg)
        # INPUT_DIR is ./input
        file_path = os.path.join(INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {file_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.is_test:
            # Return image and ID for submission mapping
            return image, row["image_id"]
        else:
            # Get label index (0-3)
            # We assume one-hot encoding or similar in the CSV, finding the max column
            # CLASS_LABELS order determines the index: ["healthy", "multiple_diseases", "rust", "scab"]
            labels_one_hot = row[CLASS_LABELS].values.astype(float)
            label_index = np.argmax(labels_one_hot)

            return image, torch.tensor(label_index, dtype=torch.long)


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed transformations.
    """
    # Common normalization stats (ImageNet defaults are usually fine,
    # but here we use standard 0-1 scaling implicitly via Normalize default or specific stats if needed.
    # We will use standard ImageNet mean/std for transfer learning compatibility)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                # Strong Geometric Augmentations as per strategy
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Explicitly excluding Cutout/CoarseDropout to preserve lesions
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_dataloaders(debug=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        debug (bool): If True, subsamples the datasets for quick debugging.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(SEED)

    # Load Metadata
    if (
        not os.path.exists(TRAIN_CSV)
        or not os.path.exists(VAL_CSV)
        or not os.path.exists(TEST_CSV)
    ):
        raise FileNotFoundError(
            "Metadata CSVs not found. Please ensure metadata generation was successful."
        )

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Debug mode: subsample
    if debug:
        train_df = train_df.head(batch_size * 2)
        val_df = val_df.head(batch_size)
        test_df = test_df.head(batch_size)
        print(
            f"[DEBUG] Subsampled data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )

    # Initialize Datasets
    train_dataset = AppleDataset(
        train_df, transforms=get_transforms("train"), is_test=False
    )

    val_dataset = AppleDataset(val_df, transforms=get_transforms("val"), is_test=False)

    test_dataset = AppleDataset(
        test_df, transforms=get_transforms("test"), is_test=True
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
