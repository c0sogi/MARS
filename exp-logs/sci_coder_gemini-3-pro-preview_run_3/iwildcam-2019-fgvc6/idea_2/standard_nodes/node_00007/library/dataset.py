import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


class AnimalDataset(Dataset):
    """
    Custom Dataset for Animal Classification.
    Reads images via OpenCV, applies Albumentations transforms.
    """

    def __init__(self, df, root_dir, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'Id', 'file_path', and optionally 'Category'.
            root_dir (str): Root directory where images are stored (e.g., ./input).
            transform (albumentations.Compose): Transforms to apply.
            is_test (bool): If True, returns dummy label.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["Id"]

        # Construct full file path
        # Metadata file_path is relative (e.g., 'train_images/xxx.jpg')
        # root_dir is usually './input'
        file_path = os.path.join(self.root_dir, row["file_path"])

        # Read image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing images (should be caught by metadata check, but for safety)
            # Create a black image of expected size
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Get label
        if self.is_test:
            label = -1  # Dummy label for test set
        else:
            label = row["Category"]

        return image, torch.tensor(label, dtype=torch.long), image_id


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders():
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    Handles WeightedRandomSampler for training to address class imbalance.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debug Mode: Sample a subset
    if Config.DEBUG:
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # --- Training Sampler (WeightedRandomSampler) ---
    # Calculate weights for each class to balance sampling
    class_counts = train_df["Category"].value_counts().sort_index()
    # Handle missing classes in the subset if any (mostly relevant for DEBUG)
    # We map existing classes to weights, others get 0 or handled safely

    # Weight = 1 / Frequency
    class_weights = 1.0 / class_counts

    # Assign a weight to each sample in the dataframe
    # map series index (category) to weight
    sample_weights = train_df["Category"].map(class_weights)

    # Convert to tensor
    sample_weights = torch.from_numpy(sample_weights.values).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # --- Datasets ---
    train_dataset = AnimalDataset(
        df=train_df,
        root_dir=Config.INPUT_ROOT,
        transform=get_transforms("train"),
        is_test=False,
    )

    val_dataset = AnimalDataset(
        df=val_df,
        root_dir=Config.INPUT_ROOT,
        transform=get_transforms("val"),
        is_test=False,
    )

    test_dataset = AnimalDataset(
        df=test_df,
        root_dir=Config.INPUT_ROOT,
        transform=get_transforms("test"),
        is_test=True,
    )

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Sampler is mutually exclusive with shuffle
        shuffle=False,
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
