import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from library.config import Config
from library.utils import set_seed


class AnimalDataset(Dataset):
    """
    Custom Dataset for Animal Classification.
    Reads images based on metadata paths and applies transforms.
    """

    def __init__(self, metadata, transform=None, is_test=False):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing 'file_path' and 'Category'/'Id'.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): If True, returns (image, id). If False, returns (image, label).
        """
        self.metadata = metadata
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Construct full file path
        # Metadata 'file_path' is relative, e.g., "train_images/img.jpg"
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(img_path)

        # Handle potential loading errors (though metadata verification suggests paths are valid)
        if image is None:
            # Return a black image of correct size to prevent crashing
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms compatibility
        image = Image.fromarray(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # For test set, we need the ID to create the submission file
            return image, str(row["Id"])
        else:
            # For train/val, we need the class label
            label = torch.tensor(row["Category"], dtype=torch.long)
            return image, label


def get_transforms(split="train"):
    """
    Generates the transformation pipeline based on the data split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The composed transforms.
    """
    transform_list = []

    # 1. Resize to native resolution of EfficientNet-B3 (300x300)
    transform_list.append(transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)))

    # 2. Minimal Augmentation for Training
    if split == "train":
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    # 3. Convert to Tensor
    transform_list.append(transforms.ToTensor())

    # 4. Normalize using ImageNet statistics
    transform_list.append(transforms.Normalize(mean=Config.MEAN, std=Config.STD))

    return transforms.Compose(transform_list)


def get_dataloaders(debug=None, batch_size=None, num_workers=None):
    """
    Creates DataLoaders for training, validation, and testing.

    Args:
        debug (bool): If True, subsets the data for rapid prototyping.
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if debug is None:
        debug = Config.DEBUG
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    set_seed()

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debug Sampling
    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"DEBUG Mode Active: Training on {len(train_df)} samples.")

    # Define Transforms
    train_transform = get_transforms(split="train")
    eval_transform = get_transforms(
        split="val"
    )  # Validation and Test use the same transforms

    # Instantiate Datasets
    train_dataset = AnimalDataset(train_df, transform=train_transform, is_test=False)
    val_dataset = AnimalDataset(val_df, transform=eval_transform, is_test=False)
    test_dataset = AnimalDataset(test_df, transform=eval_transform, is_test=True)

    # Configure DataLoader settings
    pin_memory = Config.DEVICE == "cuda"

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batch to maintain statistics stability
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
