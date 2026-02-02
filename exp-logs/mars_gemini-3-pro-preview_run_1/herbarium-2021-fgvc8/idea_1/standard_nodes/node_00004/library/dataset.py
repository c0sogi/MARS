import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformations for the specified mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=256, width=256),
                A.RandomCrop(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_label_mapping(train_df, load_cached_data=True):
    """
    Generates or loads a mapping from category_id to model output index (0..N-1).
    Caches the unique categories array to a .npy file.
    """
    cache_path = os.path.join(Config.IDEA_DIR, "classes.npy")

    # Ensure directory exists
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        unique_cats = np.load(cache_path)
    else:
        # Compute unique categories from the full training set
        unique_cats = np.sort(train_df["category_id"].unique())
        np.save(cache_path, unique_cats)

    # Create dictionary mapping: category_id -> index
    label_map = {cat_id: idx for idx, cat_id in enumerate(unique_cats)}
    return label_map, unique_cats


class HerbariumDataset(Dataset):
    def __init__(self, df, transform=None, label_map=None, is_test=False):
        self.df = df
        self.transform = transform
        self.label_map = label_map
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative to input directory (e.g., "train/images/...")
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for corrupt/missing images (should be rare given metadata check)
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.is_test:
            # Return image and image_id for submission
            return image, row["image_id"]
        else:
            # Return image and mapped label
            cat_id = row["category_id"]
            label = self.label_map[cat_id]
            return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Ensure label mapping is consistent with the full training set
    # We load/compute this using the full train_df before any debug sampling
    label_map, _ = get_label_mapping(train_df, load_cached_data=True)

    if debug:
        train_df = train_df.sample(
            n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)

    # Create Datasets
    train_dataset = HerbariumDataset(
        train_df, transform=get_transforms("train"), label_map=label_map, is_test=False
    )

    val_dataset = HerbariumDataset(
        val_df, transform=get_transforms("val"), label_map=label_map, is_test=False
    )

    test_dataset = HerbariumDataset(
        test_df, transform=get_transforms("test"), label_map=None, is_test=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
