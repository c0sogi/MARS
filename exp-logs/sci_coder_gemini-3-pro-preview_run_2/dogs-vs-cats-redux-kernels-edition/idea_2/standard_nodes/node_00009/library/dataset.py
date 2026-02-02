import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything


def get_transforms(mode="train"):
    """
    Returns the torchvision transformations for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The composition of transforms.
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    Config.IMG_SIZE, scale=Config.AUG_CROP_SCALE
                ),
                transforms.RandomHorizontalFlip(p=Config.AUG_HFLIP_PROB),
                transforms.ColorJitter(
                    brightness=Config.AUG_COLOR_JITTER,
                    contrast=Config.AUG_COLOR_JITTER,
                    saturation=Config.AUG_COLOR_JITTER,
                    hue=Config.AUG_COLOR_JITTER,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
            ]
        )
    else:
        # Validation and Test
        return transforms.Compose(
            [
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
            ]
        )


class DogCatDataset(Dataset):
    """
    Custom Dataset for Dog vs Cat classification.
    """

    def __init__(self, df, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

        # Pre-calculate full paths to avoid overhead in __getitem__
        self.filepaths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in self.df["filepath"]
        ]

        if self.mode != "test":
            self.labels = self.df["label"].values
        else:
            self.ids = self.df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.filepaths[idx]

        # Open image and convert to RGB (PIL handles this gracefully)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt images if any, though dataset is assumed clean
            print(f"Error loading image {path}: {e}")
            # Return a black image or handle appropriately.
            # Here we just create a blank image to avoid crashing the worker
            image = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        if self.mode == "test":
            # Return image and ID for submission mapping
            return image, self.ids[idx]
        else:
            # Return image and label (float for BCEWithLogitsLoss)
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label


def get_fold_dataloaders(fold_idx, n_folds=Config.N_FOLDS):
    """
    Creates DataLoaders for a specific fold in K-Fold Cross Validation.
    Merges original train and val sets to maximize data utility.

    Args:
        fold_idx (int): The index of the fold to use for validation (0 to n_folds-1).
        n_folds (int): Total number of folds.

    Returns:
        tuple: (train_loader, val_loader)
    """
    seed_everything(Config.SEED)

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_CSV)
    val_df = pd.read_csv(Config.VAL_METADATA_CSV)

    # Combine datasets for K-Fold
    full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    # Create Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    # Get indices for the requested fold
    # skf.split returns a generator, we iterate to find the specific fold
    for i, (train_index, val_index) in enumerate(skf.split(full_df, full_df["label"])):
        if i == fold_idx:
            fold_train_df = full_df.iloc[train_index]
            fold_val_df = full_df.iloc[val_index]
            break
    else:
        raise ValueError(f"Fold index {fold_idx} out of range for {n_folds} folds.")

    # Create Datasets
    train_dataset = DogCatDataset(
        fold_train_df, mode="train", transform=get_transforms("train")
    )
    val_dataset = DogCatDataset(
        fold_val_df, mode="val", transform=get_transforms("val")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
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

    return train_loader, val_loader


def get_test_dataloader():
    """
    Creates DataLoader for the test set.

    Returns:
        DataLoader: Test data loader.
    """
    test_df = pd.read_csv(Config.TEST_METADATA_CSV)

    test_dataset = DogCatDataset(test_df, mode="test", transform=get_transforms("test"))

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
