import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"


class RetinopathyDataset(Dataset):
    """
    Custom Dataset for Diabetic Retinopathy Classification.
    Handles image loading, resizing, and ordinal target generation.
    """

    def __init__(self, csv_file, transforms=None, mode="train", root_dir=INPUT_DIR):
        """
        Args:
            csv_file (str): Path to the metadata CSV file.
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
            root_dir (str): Root directory for images.
        """
        self.df = pd.read_csv(csv_file)
        self.transforms = transforms
        self.mode = mode
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative to input dir, e.g., "train_images/xxxx.png"
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(img_path)

        # Handle potential missing images (though metadata is validated)
        if image is None:
            # Return a blank image to prevent crash, though this should not happen
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode == "test":
            # For test, return image and id_code for submission
            return image, row["id_code"]
        else:
            # For train/val, return image and ordinal targets
            label = row["diagnosis"]

            # Ordinal Regression Target Encoding
            # We have 5 classes (0-4), so we create 4 binary tasks.
            # Task k checks if label > k.
            # Label 0: [0, 0, 0, 0]
            # Label 1: [1, 0, 0, 0]
            # Label 2: [1, 1, 0, 0]
            # Label 3: [1, 1, 1, 0]
            # Label 4: [1, 1, 1, 1]
            target = torch.zeros(4, dtype=torch.float32)
            for i in range(4):
                if label > i:
                    target[i] = 1.0

            return image, target


def get_transforms(image_size=512, mode="train"):
    """
    Returns the Albumentations transforms for the given mode.

    Args:
        image_size (int): Target image size (height and width).
        mode (str): 'train' or 'val'/'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                # "Squashing" resize: Resize directly to target dims, ignoring aspect ratio
                A.Resize(height=image_size, width=image_size),
                # Strict Geometric Invariance (No Hue/Saturation shifts)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Normalization (ImageNet defaults)
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # "Squashing" resize
                A.Resize(height=image_size, width=image_size),
                # Normalization (ImageNet defaults)
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    batch_size=16, image_size=512, num_workers=8, metadata_dir=METADATA_DIR
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        image_size (int): Image resolution.
        num_workers (int): Number of workers for DataLoader.
        metadata_dir (str): Directory containing metadata CSVs.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Define Transforms
    train_transforms = get_transforms(image_size, mode="train")
    val_transforms = get_transforms(image_size, mode="val")

    # Initialize Datasets
    train_dataset = RetinopathyDataset(
        train_csv, transforms=train_transforms, mode="train"
    )
    val_dataset = RetinopathyDataset(val_csv, transforms=val_transforms, mode="val")
    test_dataset = RetinopathyDataset(test_csv, transforms=val_transforms, mode="test")

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last to maintain consistent batch statistics
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
