import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
    SEED,
)
from library.utils import seed_everything

# ImageNet Normalization Constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class DogDataset(Dataset):
    """
    Custom Dataset for loading Dog images.
    """

    def __init__(self, df, root_dir, transform=None, is_test=False):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct full image path
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image and convert to RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt images (unlikely in this dataset but good practice)
            print(f"Warning: Could not load image {img_path}. Error: {e}")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Return data based on mode
        if self.is_test:
            return image, row["id"]
        else:
            return image, torch.tensor(row["label_idx"], dtype=torch.long)


def get_transforms(phase="train"):
    """
    Returns torchvision transforms for the specified phase.
    """
    if phase == "train":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        # Validation and Test
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )


def process_data(
    csv_path, cache_name, class_to_idx=None, is_test=False, load_cached_data=True
):
    """
    Loads metadata, processes it (mapping labels), and handles caching.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Construct cache path
    # Include DEBUG in filename to prevent caching debug data for full runs
    suffix = "_debug" if DEBUG else ""
    cache_path = os.path.join(WORKING_DIR, f"{cache_name}{suffix}.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute data from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Apply Debugging Limit
    if DEBUG:
        df = df.head(DEBUG_SAMPLE_SIZE)

    # Process Labels for Train/Val
    if not is_test:
        if class_to_idx is None:
            raise ValueError(
                "class_to_idx must be provided for training/validation data."
            )

        # Map breed names to integers
        df["label_idx"] = df["breed"].map(class_to_idx)

        # Verify mapping
        if df["label_idx"].isnull().any():
            raise ValueError(
                "Some breeds in the dataset could not be mapped to indices."
            )

    # Save to cache
    df.to_parquet(cache_path, index=False)

    return df


def get_class_mapping(train_csv_path):
    """
    Generates a consistent mapping from breed name to integer index
    based on the full training dataset.
    """
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"{train_csv_path} not found.")

    df = pd.read_csv(train_csv_path)
    classes = sorted(df["breed"].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return class_to_idx, classes


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare datasets and dataloaders.
    """
    # Ensure reproducibility
    seed_everything(SEED)

    # 1. Generate Class Mapping (always from full train set for consistency)
    class_to_idx, classes = get_class_mapping(TRAIN_CSV)

    # 2. Process Metadata (Load/Cache)
    train_df = process_data(
        TRAIN_CSV,
        "train_processed",
        class_to_idx=class_to_idx,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    val_df = process_data(
        VAL_CSV,
        "val_processed",
        class_to_idx=class_to_idx,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    test_df = process_data(
        TEST_CSV, "test_processed", is_test=True, load_cached_data=load_cached_data
    )

    # 3. Create Dataset Objects
    train_dataset = DogDataset(
        train_df, INPUT_DIR, transform=get_transforms("train"), is_test=False
    )
    val_dataset = DogDataset(
        val_df, INPUT_DIR, transform=get_transforms("val"), is_test=False
    )
    test_dataset = DogDataset(
        test_df, INPUT_DIR, transform=get_transforms("test"), is_test=True
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, classes
