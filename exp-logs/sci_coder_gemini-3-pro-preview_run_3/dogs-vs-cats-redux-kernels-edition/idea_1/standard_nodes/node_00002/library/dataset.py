import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the torchvision transformations for the given mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The composition of transforms.
    """
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Baseline strategy: Resize -> ToTensor -> Normalize
    transforms_list = [
        T.ToPILImage(),
        T.Resize(Config.IMAGE_SIZE),
    ]

    # Add augmentation for training
    if mode == "train":
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))

    transforms_list.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    return T.Compose(transforms_list)


class DogCatDataset(Dataset):
    """
    Custom Dataset for loading Dog vs Cat images.
    """

    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            transform (callable, optional): Transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'. Determines return values.
        """
        self.df = df
        self.transform = transform
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata filepath is relative to input dir (e.g., "train/cat.0.jpg")
        img_path = os.path.join(self.input_dir, row["filepath"])

        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for corrupt images (though metadata check passed)
            # Return a black image or raise error. Here we raise error to be safe.
            raise FileNotFoundError(f"Image not found or corrupt: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Return data based on mode
        if self.mode == "test":
            # For test, we need the ID for submission
            img_id = row["id"]
            return image, img_id
        else:
            # For train/val, we need the label
            # BCEWithLogitsLoss expects float target.
            # We return shape (1,) to match model output (N, 1) usually,
            # or scalar and unsqueeze in loop. Here we return scalar tensor.
            label = row["label"]
            return image, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(debug=False):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        debug (bool): If True, uses a small subset of data for debugging.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug Mode: Subsample data
    if debug:
        train_df = train_df.head(Config.BATCH_SIZE * 2)
        val_df = val_df.head(Config.BATCH_SIZE)
        test_df = test_df.head(Config.BATCH_SIZE)

    # Define Transforms
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")
    test_transform = get_transforms(mode="test")

    # Create Datasets
    train_dataset = DogCatDataset(train_df, transform=train_transform, mode="train")
    val_dataset = DogCatDataset(val_df, transform=val_transform, mode="val")
    test_dataset = DogCatDataset(test_df, transform=test_transform, mode="test")

    # Create DataLoaders
    # Pin memory is generally recommended for GPU training
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
