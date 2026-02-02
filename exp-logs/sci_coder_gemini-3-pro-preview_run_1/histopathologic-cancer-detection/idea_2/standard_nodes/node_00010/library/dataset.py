import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(phase: str):
    """
    Returns the Albumentations transformation pipeline for the specified phase.

    Args:
        phase (str): One of 'train', 'valid', 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    # Standard ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Hard Attention: Crop center 48x48
    # This captures the 32x32 ROI plus a small context margin
    crop_size = Config.INPUT_SIZE

    if phase == "train":
        return A.Compose(
            [
                A.CenterCrop(height=crop_size, width=crop_size),
                A.HorizontalFlip(p=Config.AUGMENTATION_PARAMS["horizontal_flip_prob"]),
                A.VerticalFlip(p=Config.AUGMENTATION_PARAMS["vertical_flip_prob"]),
                A.RandomRotate90(p=Config.AUGMENTATION_PARAMS["rotate_90_prob"]),
                # Conservative Color Jitter: Weak brightness/contrast, no hue/saturation changes
                # to avoid distorting stain biomarkers.
                A.ColorJitter(
                    brightness=Config.AUGMENTATION_PARAMS["brightness_limit"],
                    contrast=Config.AUGMENTATION_PARAMS["contrast_limit"],
                    saturation=0,
                    hue=0,
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Deterministic cropping and normalization
        return A.Compose(
            [
                A.CenterCrop(height=crop_size, width=crop_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class PathologyDataset(Dataset):
    """
    Custom Dataset for Digital Pathology Tumor Detection.
    """

    def __init__(self, metadata_df, transform=None, phase="train"):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'id', 'file_path', and 'label' (optional).
            transform (callable, optional): Albumentations transform pipeline.
            phase (str): 'train', 'valid', or 'test'.
        """
        self.metadata = metadata_df
        self.transform = transform
        self.phase = phase
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        rel_path = row["file_path"]

        # Construct full path
        img_path = os.path.join(self.input_dir, rel_path)

        # Load image using OpenCV
        image = cv2.imread(img_path)

        # Handle cases where image might not load correctly
        if image is None:
            # Create a blank image or raise error.
            # For robustness in training, we create a black image of original size
            image = np.zeros(
                (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE, 3), dtype=np.uint8
            )
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Prepare output
        if self.phase in ["train", "valid"]:
            label = row["label"]
            # Return float tensor for BCEWithLogitsLoss
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            # Test phase: return image and id for submission
            img_id = row["id"]
            return image, img_id


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, subsets the data for rapid prototyping.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debug Subsampling
    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Create Datasets
    train_dataset = PathologyDataset(
        train_df, transform=get_transforms("train"), phase="train"
    )
    val_dataset = PathologyDataset(
        val_df, transform=get_transforms("valid"), phase="valid"
    )
    test_dataset = PathologyDataset(
        test_df, transform=get_transforms("test"), phase="test"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to maintain stable statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
