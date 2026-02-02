import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_transforms


class AppleDataset(Dataset):
    """
    Custom Dataset for Apple Disease Detection.
    Handles image loading, transformation, and multi-hot label encoding.
    """

    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
        self.class_labels = Config.CLASS_LABELS
        # Create a mapping from label string to index
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.class_labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image"]

        # Construct full image path
        # metadata file_path is relative (e.g., "train_images/abc.jpg")
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(full_path)
        if image is None:
            # Handle missing images gracefully (though analysis showed 0 missing)
            # Return a blank image to prevent crashing
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback: Convert to tensor (C, H, W) and float
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Process Labels (Multi-Hot Encoding)
        # Labels are space-delimited strings in the CSV
        label_str = str(row["labels"])
        labels = label_str.split()

        # Initialize zero vector
        target = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)

        # Set 1.0 for present classes
        for lbl in labels:
            if lbl in self.label_to_idx:
                target[self.label_to_idx[lbl]] = 1.0

        return image, target, image_id


def get_loaders(debug_sample_size=None):
    """
    Initializes and returns DataLoaders for train, validation, and test sets.

    Args:
        debug_sample_size (int, optional): If provided, limits the dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata DataFrames
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Apply Debug Sampling if requested
    if debug_sample_size is not None:
        train_df = train_df.head(debug_sample_size)
        val_df = val_df.head(debug_sample_size)
        test_df = test_df.head(debug_sample_size)

    # Retrieve Transforms
    train_transforms = get_transforms(data_type="train")
    # Validation and Test use the same deterministic transforms
    val_transforms = get_transforms(data_type="valid")

    # Instantiate Datasets
    train_dataset = AppleDataset(train_df, transforms=train_transforms)
    val_dataset = AppleDataset(val_df, transforms=val_transforms)
    test_dataset = AppleDataset(test_df, transforms=val_transforms)

    # Instantiate DataLoaders
    # Drop last batch in training to maintain batch statistics consistency
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

    return train_loader, val_loader, test_loader
