import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import load_hierarchy_mappings


class PlantDataset(Dataset):
    """
    Dataset class for Plant Classification.
    Handles loading images, applying transforms, and returning hierarchical labels.
    """

    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transform (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transform = transform
        self.mode = mode

        # Pre-fetch paths and IDs to minimize overhead in __getitem__
        self.file_paths = df["file_path"].values
        self.image_ids = df["image_id"].values

        # Pre-fetch labels for training/validation
        if self.mode != "test":
            self.species_ids = df["category_id"].values

            # Ensure hierarchy columns exist (merged in get_dataloaders)
            if "genus_id" in df.columns and "family_id" in df.columns:
                self.genus_ids = df["genus_id"].values
                self.family_ids = df["family_id"].values
            else:
                # Fallback if hierarchy not merged (safety)
                self.genus_ids = np.zeros_like(self.species_ids)
                self.family_ids = np.zeros_like(self.species_ids)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full path
        file_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, file_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            # Handle missing/corrupt images by returning a black image
            # This prevents the dataloader from crashing during long runs
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode == "test":
            # Return image and ID for submission generation
            return image, self.image_ids[idx]
        else:
            # Return image and hierarchical labels dictionary
            labels = {
                "species": torch.tensor(self.species_ids[idx], dtype=torch.long),
                "genus": torch.tensor(self.genus_ids[idx], dtype=torch.long),
                "family": torch.tensor(self.family_ids[idx], dtype=torch.long),
            }
            return image, labels


def get_transforms(mode="train", img_size=224):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): 'train' or 'val'/'test'.
        img_size (int): Target image resolution.
    """
    if mode == "train":
        # Strong Data Augmentation for training
        return A.Compose(
            [
                A.RandomResizedCrop(
                    height=img_size,
                    width=img_size,
                    scale=Config.AUG_SCALE,
                    ratio=Config.AUG_RATIO,
                ),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=Config.COLOR_JITTER,
                    contrast=Config.COLOR_JITTER,
                    saturation=Config.COLOR_JITTER,
                    hue=0.1,
                    p=0.8,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test transforms (Resize + Normalize)
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(img_size, batch_size, debug=False):
    """
    Constructs DataLoaders for train, validation, and test sets.
    Merges taxonomic hierarchy information into the datasets.

    Args:
        img_size (int): Image resolution.
        batch_size (int): Batch size.
        debug (bool): If True, subsets data for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Hierarchy Mappings (Cached)
    # This uses the utility function to load or compute hierarchy mappings
    hierarchy_df = load_hierarchy_mappings(
        Config.TRAIN_METADATA_JSON, Config.HIERARCHY_CACHE_PATH, load_cached_data=True
    )

    # 2. Load Metadata CSVs
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 3. Debug Subsampling
    if debug:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 4. Merge Hierarchy Info
    # Only for train and val, as test does not have category_id
    train_df = train_df.merge(hierarchy_df, on="category_id", how="left")
    val_df = val_df.merge(hierarchy_df, on="category_id", how="left")

    # Fill NaN values in hierarchy (if any) and cast to int
    train_df["genus_id"] = train_df["genus_id"].fillna(0).astype(int)
    train_df["family_id"] = train_df["family_id"].fillna(0).astype(int)
    val_df["genus_id"] = val_df["genus_id"].fillna(0).astype(int)
    val_df["family_id"] = val_df["family_id"].fillna(0).astype(int)

    # 5. Create Datasets
    train_ds = PlantDataset(
        train_df, transform=get_transforms("train", img_size), mode="train"
    )
    val_ds = PlantDataset(val_df, transform=get_transforms("val", img_size), mode="val")
    test_ds = PlantDataset(
        test_df, transform=get_transforms("val", img_size), mode="test"
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
