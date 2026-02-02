import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config


def get_transforms(data_type, cfg):
    """
    Returns the torchvision transforms for the given data split.
    Cite solution_lesson_node_00005: Automated Augmentation Policies (RandAugment).
    """
    if data_type == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(cfg.IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(cfg.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
            ]
        )


class CassavaDataset(Dataset):
    """
    Custom Dataset for Cassava Leaf Disease Classification.
    """

    def __init__(self, df, input_dir, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            input_dir (str): Root directory for input data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Load image using PIL
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get label
        label = torch.tensor(row["label"], dtype=torch.long)

        return image, label


def get_dataloaders(cfg):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        cfg (Config): Configuration object instance.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata CSVs
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    val_df = pd.read_csv(cfg.VAL_CSV)
    test_df = pd.read_csv(cfg.TEST_CSV)

    # Handle Debug Mode
    if cfg.DEBUG:
        train_df = train_df.head(cfg.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(cfg.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(cfg.DEBUG_SAMPLE_SIZE)

    # Define Transforms
    train_transform = get_transforms("train", cfg)
    val_transform = get_transforms("val", cfg)
    test_transform = get_transforms("test", cfg)

    # Instantiate Datasets
    train_dataset = CassavaDataset(
        df=train_df, input_dir=cfg.INPUT_DIR, transform=train_transform
    )

    val_dataset = CassavaDataset(
        df=val_df, input_dir=cfg.INPUT_DIR, transform=val_transform
    )

    test_dataset = CassavaDataset(
        df=test_df, input_dir=cfg.INPUT_DIR, transform=test_transform
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to maintain batch statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
