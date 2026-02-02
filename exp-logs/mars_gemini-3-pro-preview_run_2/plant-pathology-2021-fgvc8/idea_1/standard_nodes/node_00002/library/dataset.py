import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library import config


# Set random seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(config.SEED)


class AppleDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        Custom Dataset for Apple Disease Detection.

        Args:
            df (pd.DataFrame): DataFrame containing metadata (image, labels, file_path).
            transforms (albumentations.Compose): Image transformations.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.classes = config.CLASSES
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Pre-process labels for training and validation
        if self.mode != "test":
            self.labels = self._process_labels()
        else:
            # Placeholder for test set
            self.labels = np.zeros((len(self.df), len(self.classes)), dtype=np.float32)

    def _process_labels(self):
        """
        Converts space-delimited label strings into multi-hot binary vectors.
        """
        label_matrix = np.zeros((len(self.df), len(self.classes)), dtype=np.float32)
        for idx, label_str in enumerate(self.df["labels"]):
            if not isinstance(label_str, str):
                continue
            current_labels = label_str.split()
            for lbl in current_labels:
                if lbl in self.class_to_idx:
                    class_id = self.class_to_idx[lbl]
                    label_matrix[idx, class_id] = 1.0
        return label_matrix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image"]

        # Construct full file path
        file_path = os.path.join(config.INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {file_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Default transform if none provided
            transform = A.Compose(
                [
                    A.Resize(config.IMG_SIZE, config.IMG_SIZE),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            augmented = transform(image=image)
            image = augmented["image"]

        target = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.mode == "test":
            # Return image_id for submission generation
            return image, target, image_id
        else:
            return image, target


def get_transforms(mode="train", img_size=config.IMG_SIZE):
    """
    Returns the image augmentation pipeline.

    Args:
        mode (str): 'train' or 'val'/'test'.
        img_size (int): Target image size.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=img_size // 10,
                    max_width=img_size // 10,
                    p=0.3,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    train_batch_size=config.BATCH_SIZE,
    val_batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    img_size=config.IMG_SIZE,
):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        train_batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation/testing.
        num_workers (int): Number of worker threads for data loading.
        img_size (int): Image size for resizing.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(config.TRAIN_META_PATH)
    val_df = pd.read_csv(config.VAL_META_PATH)
    test_df = pd.read_csv(config.TEST_META_PATH)

    # Define Transforms
    train_transforms = get_transforms(mode="train", img_size=img_size)
    val_transforms = get_transforms(mode="val", img_size=img_size)
    test_transforms = get_transforms(mode="test", img_size=img_size)

    # Initialize Datasets
    train_dataset = AppleDataset(train_df, transforms=train_transforms, mode="train")
    val_dataset = AppleDataset(val_df, transforms=val_transforms, mode="val")
    test_dataset = AppleDataset(test_df, transforms=test_transforms, mode="test")

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
