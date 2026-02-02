import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class AppleLeafDataset(Dataset):
    """
    PyTorch Dataset for Apple Leaf Disease Detection.
    Reads images from disk and applies augmentations.
    """

    def __init__(self, df, transforms=None, root_dir=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image_id, file_path, and labels.
            transforms (albumentations.Compose): Albumentations transforms to apply.
            root_dir (str): Root directory for images. Defaults to Config.INPUT_DIR.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.root_dir = root_dir if root_dir else Config.INPUT_DIR
        self.classes = Config.CLASSES

        # Check if labels exist in the dataframe
        self.has_labels = all(c in self.df.columns for c in self.classes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative path (e.g., "images/Train_0.jpg")
        # Config.INPUT_DIR is "./input"
        file_path = os.path.join(self.root_dir, row["file_path"])

        # Read image
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {file_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to tensor conversion if no transforms provided
            image = ToTensorV2()(image=image)["image"]

        # Prepare return dictionary
        result = {"image": image, "image_id": row["image_id"]}

        # Add labels if available
        if self.has_labels:
            # Extract labels in the order defined in Config.CLASSES
            labels = row[self.classes].values.astype(np.float32)
            result["target"] = torch.tensor(labels, dtype=torch.float32)

        return result


def get_transforms(phase, img_size):
    """
    Creates the Albumentations transform pipeline.

    Args:
        phase (str): 'train' or 'valid'/'test'.
        img_size (int): Target image size (height and width).

    Returns:
        albumentations.Compose: The transform pipeline.
    """
    if phase == "train":
        return A.Compose(
            [
                # Strong Geometric Augmentations
                # Explicitly excluding occlusion (Cutout, CoarseDropout) as per strategy
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=Config.AUG_SHIFT_LIMIT,
                    scale_limit=Config.AUG_SCALE_LIMIT,
                    rotate_limit=Config.AUG_ROTATE_LIMIT,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.7,
                ),
                # Resize to target resolution
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / Inference
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )


def get_loaders(img_size, batch_size=None, debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        img_size (int): The resolution for images (e.g., 224 or 300).
        batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
        debug (bool): If True, uses a small subset of data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Paths to metadata
    train_csv = os.path.join(Config.METADATA_DIR, "train.csv")
    val_csv = os.path.join(Config.METADATA_DIR, "val.csv")
    test_csv = os.path.join(Config.METADATA_DIR, "test.csv")

    # Load DataFrames
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Debug mode: subsample data
    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Create Transforms
    train_transforms = get_transforms("train", img_size)
    val_transforms = get_transforms("valid", img_size)

    # Create Datasets
    train_dataset = AppleLeafDataset(train_df, transforms=train_transforms)
    val_dataset = AppleLeafDataset(val_df, transforms=val_transforms)
    test_dataset = AppleLeafDataset(test_df, transforms=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
