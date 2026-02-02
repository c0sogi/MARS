import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import ordinal_encode


def get_transforms(phase: str):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Args:
        phase (str): One of 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    # Base transforms: Resize and Normalize
    transforms_list = [
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    # Add augmentations for training
    if phase == "train":
        transforms_list.insert(1, A.HorizontalFlip(p=0.5))
        # Additional lightweight augmentations can be added here if needed
        # e.g., A.RandomBrightnessContrast(p=0.2)

    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


class RetinopathyDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy classification.
    Handles image loading, preprocessing, and ordinal label encoding.
    """

    def __init__(self, df: pd.DataFrame, phase: str = "train", transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id_code, file_path, diagnosis).
            phase (str): 'train', 'val', or 'test'.
            transform (A.Compose, optional): Albumentations transform pipeline.
        """
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.transform = transform

        # Pre-check columns
        self.has_diagnosis = "diagnosis" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative (e.g., "train_images/xxxx.png")
        # Config.INPUT_DIR is "./input"
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing/corrupt images (should be rare given metadata validation)
            # Create a black image of expected size
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Handle Targets
        if self.phase != "test" and self.has_diagnosis:
            diagnosis = row["diagnosis"]
            # Convert integer label to ordinal vector
            target = ordinal_encode(diagnosis, num_classes=Config.NUM_CLASSES)
        else:
            # For test set or missing labels, return dummy target
            # Shape should match ordinal output: (NUM_CLASSES - 1,)
            target = torch.zeros(Config.NUM_ORDINAL_UNITS, dtype=torch.float32)

        return image, target


def get_dataloaders(debug_sample_size=Config.DEBUG_SAMPLE_SIZE):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug_sample_size (int, optional): If set, limits the dataset size for debugging.

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # Debug Sampling
    if debug_sample_size is not None:
        print(f"DEBUG MODE: Sampling {debug_sample_size} rows per split.")
        df_train = df_train.head(debug_sample_size)
        df_val = df_val.head(debug_sample_size)
        df_test = df_test.head(debug_sample_size)

    # Create Datasets
    train_dataset = RetinopathyDataset(
        df_train, phase="train", transform=get_transforms("train")
    )

    val_dataset = RetinopathyDataset(
        df_val, phase="val", transform=get_transforms("val")
    )

    test_dataset = RetinopathyDataset(
        df_test, phase="test", transform=get_transforms("test")
    )

    # Create DataLoaders
    # Pin memory helps with faster transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
