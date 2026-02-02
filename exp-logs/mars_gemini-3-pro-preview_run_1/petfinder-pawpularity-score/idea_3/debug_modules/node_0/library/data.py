import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.utils import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
    SEED,
    set_seed,
)

# Set seed for reproducibility
set_seed(SEED)


class PetDataset(Dataset):
    """
    PyTorch Dataset for the Pet Pawpularity Prediction task.
    Loads images and corresponding metadata/targets.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata and file paths.
            transforms (albumentations.Compose): Transformations to apply to images.
            mode (str): One of 'train', 'val', 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Metadata feature columns (binary flags)
        self.meta_features = [
            "Focus",
            "Eyes",
            "Face",
            "Near",
            "Action",
            "Accessory",
            "Group",
            "Collage",
            "Human",
            "Occlusion",
            "Info",
            "Blur",
        ]

        # Ensure file paths are complete relative to INPUT_DIR
        # The metadata csv contains 'file_path' like 'train/id.jpg'
        # We construct the full path here to save time in __getitem__
        self.file_paths = [os.path.join(INPUT_DIR, x) for x in df["file_path"].values]

        # Pre-extract metadata features as a numpy array for speed
        self.dense_features = df[self.meta_features].values.astype(np.float32)

        # Extract targets if available
        if "Pawpularity" in df.columns:
            self.targets = df["Pawpularity"].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load Image
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)

        # Handle missing or corrupt images gracefully (though dataset should be clean)
        if image is None:
            # Create a black image of expected size as fallback
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Get Metadata Features
        meta = torch.tensor(self.dense_features[idx], dtype=torch.float32)

        # Return data based on mode
        if self.mode in ["train", "val"]:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            # Return: image, metadata, target
            return image, meta, target
        else:
            # Test mode: Return image, metadata, and the ID (for submission)
            # We return the ID to ensure predictions match rows
            row_id = self.df.iloc[idx]["Id"]
            return image, meta, row_id


def get_transforms(mode="train", img_size=IMG_SIZE):
    """
    Returns the Albumentations transformations for the given mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
        img_size (int): Target image size.

    Returns:
        A.Compose: Composed transformations.
    """
    # Standard ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if mode == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                # Add slight augmentations for robustness, though linear probing
                # on strong backbones often prefers clean resizing.
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Deterministic resizing
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(),
            ]
        )


def create_dataloaders(
    train_path=TRAIN_METADATA_PATH,
    val_path=VAL_METADATA_PATH,
    test_path=TEST_METADATA_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    debug=DEBUG,
    debug_sample_size=DEBUG_SAMPLE_SIZE,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        train_path (str): Path to training metadata CSV.
        val_path (str): Path to validation metadata CSV.
        test_path (str): Path to test metadata CSV.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        debug (bool): If True, subsets the data for debugging.
        debug_sample_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load DataFrames
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Debug Mode: Subset Data
    if debug:
        print(f"Debug mode enabled. Subsetting to {debug_sample_size} samples.")
        train_df = train_df.sample(
            n=min(len(train_df), debug_sample_size), random_state=SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), debug_sample_size), random_state=SEED
        ).reset_index(drop=True)
        # We generally don't subset test set in debug unless specifically requested,
        # but for consistency in pipeline testing, we can subset if needed.
        # However, usually we want to test the full submission pipeline.
        # Here we leave test_df as is or subset if very large, but 1000 samples is small enough.

    # Define Transforms
    train_transforms = get_transforms(mode="train", img_size=IMG_SIZE)
    val_transforms = get_transforms(mode="val", img_size=IMG_SIZE)
    test_transforms = get_transforms(mode="test", img_size=IMG_SIZE)

    # Create Datasets
    train_dataset = PetDataset(train_df, transforms=train_transforms, mode="train")
    val_dataset = PetDataset(val_df, transforms=val_transforms, mode="val")
    test_dataset = PetDataset(test_df, transforms=test_transforms, mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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
