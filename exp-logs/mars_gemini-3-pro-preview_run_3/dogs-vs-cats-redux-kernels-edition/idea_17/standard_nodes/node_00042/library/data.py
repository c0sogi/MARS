import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from library.utils import seed_everything

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_17"
METADATA_DIR = "./metadata"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)


class CatDogDataset(Dataset):
    """
    Dataset class for Dog vs Cat classification.
    Handles loading images from disk based on metadata paths.
    """

    def __init__(self, df: pd.DataFrame, transform=None, mode: str = "train"):
        self.df = df
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Metadata contains relative path, e.g., "train/cat.0.jpg"
        img_path = os.path.join(INPUT_DIR, row["filepath"])

        # Open image and convert to RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupted images (though analysis showed 0 missing)
            # Return a black image or raise error depending on strictness.
            # Here we raise to fail fast if data is bad.
            raise IOError(f"Error loading image {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        if self.mode == "test":
            # Return image and ID for submission mapping
            return image, row["id"]
        else:
            # Return image and label
            label = torch.tensor(row["label"], dtype=torch.float32)
            return image, label


def get_resolution(model_name: str) -> int:
    """
    Maps model architecture to the specific input resolution required
    by the Heterogeneous Ensemble strategy.
    """
    if "resnet50" in model_name:
        return 256
    elif "convnext_small" in model_name:
        return 288
    elif "maxvit_tiny" in model_name:
        return 224
    else:
        raise ValueError(f"Unknown model name for resolution mapping: {model_name}")


def get_transforms(resolution: int, split: str):
    """
    Returns transforms based on resolution and split.
    Implements specific resizing and augmentation strategies.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if split == "train":
        return transforms.Compose(
            [
                # Context-Preserving Augmentation
                transforms.RandomResizedCrop(
                    (resolution, resolution),
                    scale=(0.8, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                # ColorJitter with intensity >= 0.2
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        # Validation / Test
        # Strict resizing to resolution using Bicubic interpolation
        return transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def create_folds(
    num_folds: int = 5, load_cached_data: bool = True, seed: int = 42
) -> pd.DataFrame:
    """
    Creates or loads stratified folds from metadata.
    Combines train.csv and val.csv to form the full dataset before splitting.
    """
    cache_path = os.path.join(CACHE_DIR, "folds.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to re-creation if load fails

    # 2. Compute data from scratch
    # Load metadata
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")

    if not os.path.exists(train_meta_path) or not os.path.exists(val_meta_path):
        raise FileNotFoundError("Metadata CSVs not found in ./metadata/")

    train_meta = pd.read_csv(train_meta_path)
    val_meta = pd.read_csv(val_meta_path)

    # Combine to use full dataset for CV
    full_df = pd.concat([train_meta, val_meta], ignore_index=True)

    # Create Stratified Folds
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    full_df["fold"] = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df["label"])):
        full_df.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    full_df.to_parquet(cache_path, index=False)

    return full_df


def get_loaders(
    fold_idx: int,
    model_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    load_cached_data: bool = True,
    seed: int = 42,
):
    """
    Returns train and validation loaders for a specific fold and model.
    """
    seed_everything(seed)

    # Get Data with Folds
    df = create_folds(num_folds=5, load_cached_data=load_cached_data, seed=seed)

    # Split based on fold index
    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    # Get Resolution based on model architecture
    resolution = get_resolution(model_name)

    # Transforms
    train_tfm = get_transforms(resolution, "train")
    val_tfm = get_transforms(resolution, "val")

    # Datasets
    train_ds = CatDogDataset(train_df, transform=train_tfm, mode="train")
    val_ds = CatDogDataset(val_df, transform=val_tfm, mode="train")

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(model_name: str, batch_size: int = 32, num_workers: int = 4):
    """
    Returns test loader for inference.
    """
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")
    if not os.path.exists(test_meta_path):
        raise FileNotFoundError("Test metadata not found.")

    df = pd.read_csv(test_meta_path)

    resolution = get_resolution(model_name)
    test_tfm = get_transforms(resolution, "test")

    test_ds = CatDogDataset(df, transform=test_tfm, mode="test")

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
