import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG
from library.utils import seed_everything


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Reads images using OpenCV and processes multi-label targets.
    """

    def __init__(self, df, transform=None):
        self.df = df
        self.file_paths = df["file_path"].values
        self.labels = df["labels"].values
        self.transform = transform
        self.class_labels = CFG.class_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        file_path = self.file_paths[idx]
        full_path = os.path.join(CFG.input_root, file_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Process labels (Multi-hot encoding)
        label_str = self.labels[idx]
        label_list = label_str.split()
        target = np.zeros(len(self.class_labels), dtype=np.float32)

        for l in label_list:
            if l in self.class_labels:
                target[self.class_labels.index(l)] = 1.0

        return image, torch.tensor(target)


def get_transforms(data="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        data (str): 'train', 'valid', or 'test'.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(CFG.img_size, CFG.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                # CoarseDropout for regularization as per Idea
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(CFG.img_size * 0.1),
                    max_width=int(CFG.img_size * 0.1),
                    min_holes=4,
                    fill_value=0,
                    p=CFG.coarse_dropout_prob,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    elif data in ["valid", "test"]:
        return A.Compose(
            [
                A.Resize(CFG.img_size, CFG.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


def get_loaders():
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    # Ensure reproducibility
    seed_everything(CFG.seed)

    # Load metadata
    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.val_csv)
    test_df = pd.read_csv(CFG.test_csv)

    # Debug mode: subset data
    if CFG.debug:
        train_df = train_df.head(100).reset_index(drop=True)
        val_df = val_df.head(50).reset_index(drop=True)
        test_df = test_df.head(50).reset_index(drop=True)

    # Create Datasets
    train_dataset = AppleDataset(train_df, transform=get_transforms(data="train"))
    val_dataset = AppleDataset(val_df, transform=get_transforms(data="valid"))
    test_dataset = AppleDataset(test_df, transform=get_transforms(data="test"))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
