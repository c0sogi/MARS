import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from library.config import (
    INPUT_DIR,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    PREFETCH_FACTOR,
    SEED,
)


class ISICDataset(Dataset):
    """
    Custom Dataset for ISIC Skin Lesion Classification.
    Loads images, applies transformations, and serves image + tabular data.
    """

    def __init__(self, df, tabular_data, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path' and 'target' (if train/val).
            tabular_data (np.ndarray): Preprocessed tabular features corresponding to df rows.
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df.reset_index(drop=True)
        self.tabular_data = tabular_data.astype(np.float32)
        self.mode = mode
        self.transform = transform

        # Pre-compute full file paths to avoid overhead in __getitem__
        # file_path in metadata is relative to INPUT_DIR (e.g., "jpeg/train/ISIC_xxxx.jpg")
        self.file_paths = [
            os.path.join(INPUT_DIR, fp) for fp in self.df["file_path"].values
        ]

        # Handle targets
        if self.mode != "test":
            self.targets = self.df["target"].values.astype(np.float32)
        else:
            # Dummy targets for test set
            self.targets = np.zeros(len(self.df), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        image_path = self.file_paths[idx]
        image = cv2.imread(image_path)

        if image is None:
            # Handle missing images gracefully
            # Create a black image of correct size
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Resize Image
        # Resize to IMG_SIZE x IMG_SIZE using OpenCV (fast)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # 3. Apply Transforms (ToTensor + Normalize)
        if self.transform:
            image = self.transform(image)
        else:
            # Fallback: Convert to tensor manually
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # 4. Get Tabular Data
        tabular = self.tabular_data[idx]

        # 5. Get Target
        target = self.targets[idx]

        return image, tabular, target


def get_dataloaders(
    train_df,
    val_df,
    test_df,
    train_tabular,
    val_tabular,
    test_tabular,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        train_df, val_df, test_df (pd.DataFrame): Metadata dataframes.
        train_tabular, val_tabular, test_tabular (np.ndarray): Preprocessed tabular features.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Define Transformations
    # ToTensor converts [0, 255] HWC image to [0.0, 1.0] CHW tensor
    # Normalize applies (image - mean) / std
    common_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Instantiate Datasets
    train_dataset = ISICDataset(
        train_df, train_tabular, mode="train", transform=common_transform
    )

    val_dataset = ISICDataset(
        val_df, val_tabular, mode="val", transform=common_transform
    )

    test_dataset = ISICDataset(
        test_df, test_tabular, mode="test", transform=common_transform
    )

    # Instantiate DataLoaders
    # drop_last=False ensures we process all samples, which is important for
    # the feature extraction phase to align with the full dataset.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
