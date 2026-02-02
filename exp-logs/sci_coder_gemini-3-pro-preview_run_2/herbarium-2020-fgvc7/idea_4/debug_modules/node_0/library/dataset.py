import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


class HerbariumDataset(Dataset):
    """
    Custom Dataset for loading Plant Species images.
    """

    def __init__(self, df, input_dir, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels (for train/val).
            input_dir (str): Root directory for input data.
            transform (albumentations.Compose): Transformations to apply to the images.
            mode (str): 'train', 'val', or 'test'. Determines what __getitem__ returns.
        """
        self.df = df
        self.input_dir = input_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # file_path in CSV is relative to input_dir (e.g., nybg2020/train/...)
        image_path = os.path.join(self.input_dir, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for missing images (though verification script showed 0 missing)
            # Create a black image to prevent crashing
            image = torch.zeros((224, 224, 3), dtype=torch.uint8).numpy()
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided: Normalize and ToTensor
            base_transform = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
            augmented = base_transform(image=image)
            image = augmented["image"]

        if self.mode in ["train", "val"]:
            label = row["category_id"]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # For test mode, return image and image_id for submission mapping
            image_id = row["image_id"]
            return image, torch.tensor(image_id, dtype=torch.long)


def get_transforms(cfg, data="train"):
    """
    Returns Albumentations transforms based on the data split.

    Args:
        cfg (Config): Configuration object.
        data (str): 'train' or 'val'/'test'.

    Returns:
        A.Compose: Composed transformations.
    """
    img_size = cfg.img_size

    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if data == "train":
        return A.Compose(
            [
                # RandomResizedCrop is standard for training Swin/ViT
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                # ShiftScaleRotate adds robustness to orientation and framing variations
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # Simple resize for validation/test
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_dataloaders(cfg):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        cfg (Config): Configuration object containing paths and params.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(cfg.train_csv_path)
    val_df = pd.read_csv(cfg.val_csv_path)
    test_df = pd.read_csv(cfg.test_csv_path)

    # Debug Mode: Subset data
    if cfg.debug:
        print(f"Debug mode active. Subsetting data to {cfg.debug_sample_size} samples.")
        train_df = train_df.head(cfg.debug_sample_size)
        val_df = val_df.head(cfg.debug_sample_size)
        test_df = test_df.head(cfg.debug_sample_size)

    # Define Transforms
    train_transform = get_transforms(cfg, data="train")
    val_transform = get_transforms(cfg, data="val")
    test_transform = get_transforms(cfg, data="test")

    # Create Datasets
    train_dataset = HerbariumDataset(
        df=train_df, input_dir=cfg.input_dir, transform=train_transform, mode="train"
    )
    val_dataset = HerbariumDataset(
        df=val_df, input_dir=cfg.input_dir, transform=val_transform, mode="val"
    )
    test_dataset = HerbariumDataset(
        df=test_df, input_dir=cfg.input_dir, transform=test_transform, mode="test"
    )

    # Create DataLoaders
    # Pin memory helps with faster data transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
