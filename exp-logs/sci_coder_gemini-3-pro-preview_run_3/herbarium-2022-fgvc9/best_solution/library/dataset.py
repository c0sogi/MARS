import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_label_mappings


class PlantDataset(Dataset):
    """
    Custom Dataset for Plant Image Categorization.
    """

    def __init__(self, df, transforms=None, mode="train", id2label=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transforms (A.Compose): Albumentations transformations.
            mode (str): 'train', 'val', or 'test'.
            id2label (dict): Mapping from original category_id to contiguous label.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.input_dir = Config.INPUT_DIR
        self.id2label = id2label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # file_path in metadata is relative to input directory
        image_path = os.path.join(self.input_dir, row["file_path"])

        # Read image using OpenCV
        image = cv2.imread(image_path)

        # Handle cases where image might not be read correctly
        if image is None:
            # Create a black image as a fallback to prevent crashing
            # In a real scenario, we might want to log this
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode == "test":
            # For test, return image and image_id (for submission)
            return image, str(row["image_id"])
        else:
            # For train/val, return image and category_id (label)
            cat_id = row["category_id"]
            if self.id2label:
                label_idx = self.id2label[cat_id]
            else:
                label_idx = cat_id
            label = torch.tensor(label_idx, dtype=torch.long)
            return image, label


def get_transforms(data_type="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data_type (str): 'train' or 'val'/'test'.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if data_type == "train":
        return A.Compose(
            [
                # Random Resized Crop for better generalization
                A.RandomResizedCrop(
                    size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE), scale=(0.8, 1.0)
                ),
                # Random Horizontal Flip
                A.HorizontalFlip(p=0.5),
                # Shift, Scale, Rotate
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                # Color Jitter
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5
                ),
                # Normalize
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                # Convert to Tensor
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                # Resize to 256x256
                A.Resize(height=256, width=256),
                # Center Crop to 224x224
                A.CenterCrop(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                # Normalize
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                # Convert to Tensor
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE,
    val_batch_size=Config.BATCH_SIZE,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        train_batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation/testing.
        debug (bool): Whether to run in debug mode (subset of data).
        debug_sample_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debug Mode: Subsample data
    if debug:
        df_train = df_train.sample(
            n=min(len(df_train), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(len(df_test), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)

    # Define Transforms
    train_transforms = get_transforms(data_type="train")
    val_transforms = get_transforms(data_type="val")

    # Generate Mappings
    id2label, _ = get_label_mappings(Config.TRAIN_METADATA_PATH)

    # Create Datasets
    train_dataset = PlantDataset(
        df_train, transforms=train_transforms, mode="train", id2label=id2label
    )
    val_dataset = PlantDataset(
        df_val, transforms=val_transforms, mode="val", id2label=id2label
    )
    test_dataset = PlantDataset(df_test, transforms=val_transforms, mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
