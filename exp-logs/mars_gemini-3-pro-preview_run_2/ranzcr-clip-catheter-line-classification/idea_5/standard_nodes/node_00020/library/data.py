import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Define input root directory
INPUT_ROOT = "./input"


class CatheterDataset(Dataset):
    """
    Dataset class for loading catheter chest x-ray images and labels.
    """

    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.target_cols = Config.target_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative (e.g., "train/ID.jpg"), so we join with input root
        img_path = os.path.join(INPUT_ROOT, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for safety, though verification ensures files exist
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode in ["train", "valid"]:
            labels = row[self.target_cols].values.astype(np.float32)
            return image, torch.tensor(labels)
        else:
            # Test mode: return image and StudyInstanceUID for submission mapping
            return image, row["StudyInstanceUID"]


def get_transforms(data="train"):
    """
    Constructs the augmentation pipeline.

    Sequence:
    1. Resize (LongestMaxSize) -> Preserves aspect ratio.
    2. CLAHE -> Enhances contrast on valid pixels (before padding).
    3. Pad (PadIfNeeded) -> Creates square 640x640 input.
    4. Augmentations (Train only) -> ShiftScaleRotate, CoarseDropout.
    5. Normalize & ToTensor.
    """
    transforms_list = [
        # Resize longest edge to 640, preserving aspect ratio
        A.LongestMaxSize(max_size=Config.image_size, always_apply=True),
        # Apply CLAHE before padding to avoid histogram skew from padding zeros
        A.CLAHE(p=1.0),
        # Pad to square 640x640
        A.PadIfNeeded(
            min_height=Config.image_size,
            min_width=Config.image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            always_apply=True,
        ),
    ]

    if data == "train":
        # Geometric and dropout augmentations for training
        transforms_list.extend(
            [
                A.ShiftScaleRotate(
                    shift_limit=Config.shift_limit,
                    scale_limit=Config.scale_limit,
                    rotate_limit=Config.rotate_limit,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=Config.aug_prob,
                ),
                A.CoarseDropout(
                    max_holes=Config.coarse_dropout_holes,
                    max_height=Config.coarse_dropout_size,
                    max_width=Config.coarse_dropout_size,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=Config.aug_prob,
                ),
            ]
        )

    # Normalization and Tensor conversion
    transforms_list.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms_list)


def get_loaders(debug=False):
    """
    Initializes and returns DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, subsets the data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load metadata CSVs
    train_df = pd.read_csv(Config.train_metadata)
    val_df = pd.read_csv(Config.val_metadata)
    test_df = pd.read_csv(Config.test_metadata)

    # Subset for debugging if requested
    if debug:
        train_df = train_df.iloc[: Config.batch_size * 2]
        val_df = val_df.iloc[: Config.valid_batch_size * 2]
        test_df = test_df.iloc[: Config.valid_batch_size * 2]

    # Instantiate Datasets
    train_dataset = CatheterDataset(
        train_df, transforms=get_transforms(data="train"), mode="train"
    )

    val_dataset = CatheterDataset(
        val_df, transforms=get_transforms(data="valid"), mode="valid"
    )

    test_dataset = CatheterDataset(
        test_df, transforms=get_transforms(data="test"), mode="test"
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
