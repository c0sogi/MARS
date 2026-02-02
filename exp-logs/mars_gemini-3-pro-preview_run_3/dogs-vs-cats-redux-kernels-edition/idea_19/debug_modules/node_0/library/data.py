import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import StratifiedKFold

from library.config import (
    INPUT_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
    SEED,
    NUM_WORKERS,
    ModelConfig,
)
from library.utils import seed_everything


class DogCatDataset(Dataset):
    """
    Custom Dataset for loading Dog vs Cat images.
    """

    def __init__(
        self, df: pd.DataFrame, root_dir: str, transform=None, is_test: bool = False
    ):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["filepath"])

        # Load image and convert to RGB (handles grayscale or RGBA)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupted images (though metadata check passed)
            # Create a black image if loading fails to prevent crash
            print(f"Warning: Failed to load {img_path}. Error: {e}")
            image = Image.new("RGB", (256, 256), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # Return image and ID for submission
            return image, row["id"]
        else:
            # Return image and label (float for BCEWithLogitsLoss)
            return image, torch.tensor(row["label"], dtype=torch.float32)


def get_transforms(cfg: ModelConfig, is_train: bool):
    """
    Generates the transformation pipeline based on the model configuration.
    Implements Resolution Diversity and Context-Preserving Augmentation.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose(
            [
                # Context-Preserving Augmentation: Scale restricted to 0.8-1.0
                transforms.RandomResizedCrop(
                    (cfg.img_size, cfg.img_size),
                    scale=(0.8, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # ColorJitter with intensity >= 0.2
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        # Validation/Test: Resize then CenterCrop to maintain aspect ratio logic
        # or direct resize if preferred. Here we use Resize -> CenterCrop
        # to ensure the subject is centered and aspect ratio is respected.
        return transforms.Compose(
            [
                transforms.Resize(
                    cfg.img_size, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(cfg.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def get_data_splits(load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads training data and assigns fold indices using Stratified K-Fold.
    Implements caching mechanism using Parquet.
    """
    cache_path = os.path.join(WORKING_DIR, "folds.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # print(f"Loaded cached folds from {cache_path}")
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    # Load metadata
    train_df = pd.read_csv(TRAIN_META_PATH)
    val_df = pd.read_csv(VAL_META_PATH)

    # Combine to form full training set (Full Data Utilization)
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    # Initialize Fold column
    full_df["fold"] = -1

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    X = np.zeros(len(full_df))
    y = full_df["label"].values

    for fold_id, (_, val_idx) in enumerate(skf.split(X, y)):
        full_df.loc[val_idx, "fold"] = fold_id

    # 3. Save to cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    full_df.to_parquet(cache_path, index=False)

    return full_df


def get_fold_loaders(
    fold_idx: int,
    cfg: ModelConfig,
    load_cached_data: bool = True,
):
    """
    Creates DataLoaders for a specific fold.

    Args:
        fold_idx: Index of the fold to use for validation (0-4).
        cfg: ModelConfig object containing architecture-specific params.
        load_cached_data: Whether to use cached fold splits.

    Returns:
        train_loader, val_loader
    """
    seed_everything(SEED)

    # Get data with fold assignments
    df = get_data_splits(load_cached_data=load_cached_data)

    # Split into Train and Validation based on fold_idx
    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    # Create Datasets
    train_dataset = DogCatDataset(
        train_df,
        root_dir=INPUT_DIR,
        transform=get_transforms(cfg, is_train=True),
        is_test=False,
    )

    val_dataset = DogCatDataset(
        val_df,
        root_dir=INPUT_DIR,
        transform=get_transforms(cfg, is_train=False),
        is_test=False,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(cfg: ModelConfig):
    """
    Creates DataLoader for the test set.
    """
    test_df = pd.read_csv(TEST_META_PATH)

    test_dataset = DogCatDataset(
        test_df,
        root_dir=INPUT_DIR,
        transform=get_transforms(cfg, is_train=False),
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
