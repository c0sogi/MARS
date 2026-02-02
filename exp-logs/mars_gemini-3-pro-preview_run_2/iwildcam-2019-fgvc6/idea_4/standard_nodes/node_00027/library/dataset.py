import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library import config


def calculate_class_weights(metadata_path=config.TRAIN_METADATA_PATH):
    """
    Calculates class weights using inverse frequency to handle class imbalance.
    Returns a torch tensor of weights on the configured device.
    """
    df = pd.read_csv(metadata_path)

    # If debugging, we might be using a subset, but weights should ideally
    # reflect the full distribution. However, if we are strictly debugging the pipeline,
    # we can calculate on the subset or just load the full csv for weights.
    # Here we load the full CSV to get accurate global weights.

    class_counts = df["Category"].value_counts().sort_index()

    # Initialize counts for all classes to handle potential missing classes in splits
    counts = np.zeros(config.NUM_CLASSES)
    for cls, count in class_counts.items():
        if 0 <= cls < config.NUM_CLASSES:
            counts[cls] = count

    # Avoid division by zero for classes that might not be present
    counts = np.maximum(counts, 1)

    total_samples = np.sum(counts)
    num_classes = len(counts)

    # Balanced weights formula: N / (num_classes * count)
    weights = total_samples / (num_classes * counts)

    return torch.FloatTensor(weights).to(config.DEVICE)


class AnimalDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing file paths and labels.
            transforms (albumentations.Compose): Transformations to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.input_dir = config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths like 'train_images/xyz.jpg'
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(file_path)

        # Handle potential missing images (though EDA showed none)
        if image is None:
            # Return a black image of correct size
            image = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB (OpenCV loads BGR, models expect RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback transform
            base_transform = A.Compose(
                [
                    A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = base_transform(image=image)["image"]

        # Return data based on mode
        if self.mode == "test":
            # For test, return image and the ID string for submission
            return image, row["Id"]
        else:
            # For train/val, return image and the class label
            return image, torch.tensor(row["Category"], dtype=torch.long)


def get_transforms(mode="train"):
    """
    Returns albumentations transforms for the specified mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                # Minimal augmentation as per strategy
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test transforms
        return A.Compose(
            [
                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders():
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    # Load Metadata
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # Handle Debug Mode
    if config.DEBUG:
        train_df = train_df.sample(
            n=min(len(train_df), config.DEBUG_SAMPLE_SIZE), random_state=config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), config.DEBUG_SAMPLE_SIZE), random_state=config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), config.DEBUG_SAMPLE_SIZE), random_state=config.SEED
        ).reset_index(drop=True)

    # Initialize Transforms
    train_transforms = get_transforms(mode="train")
    val_transforms = get_transforms(mode="val")
    test_transforms = get_transforms(mode="test")

    # Initialize Datasets
    train_dataset = AnimalDataset(train_df, transforms=train_transforms, mode="train")
    val_dataset = AnimalDataset(val_df, transforms=val_transforms, mode="val")
    test_dataset = AnimalDataset(test_df, transforms=test_transforms, mode="test")

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
