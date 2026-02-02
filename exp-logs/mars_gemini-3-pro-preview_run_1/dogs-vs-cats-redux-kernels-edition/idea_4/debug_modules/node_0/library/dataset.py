import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from library.config import CFG


def seed_worker(worker_id):
    """
    Worker initialization function to ensure reproducibility in DataLoaders.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_dataset_metadata(csv_path: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads dataset metadata from a CSV file.
    Implements caching using Parquet to satisfy deterministic processing requirements.

    Args:
        csv_path (str): Path to the source CSV file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    # Construct a cache filename based on the source filename
    filename = os.path.basename(csv_path)
    cache_name = os.path.splitext(filename)[0] + ".parquet"
    cache_path = os.path.join(CFG.working_dir, cache_name)

    # Ensure working directory exists
    os.makedirs(CFG.working_dir, exist_ok=True)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Simple validation to ensure cache isn't empty/corrupt
            if not df.empty:
                return df
        except Exception:
            # If load fails, fall through to process from scratch
            pass

    # 2. Process data from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df


def get_transforms(data: str, size: int):
    """
    Returns the Albumentations transform pipeline.

    Args:
        data (str): 'train' or 'valid'.
        size (int): Image size (height and width).
    """
    if data == "train":
        return A.Compose(
            [
                # Heavy Augmentation Pipeline
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0)),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return A.Compose(
            [
                # Deterministic validation pipeline
                A.Resize(height=size, width=size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Same as valid for test
        return A.Compose(
            [
                A.Resize(height=size, width=size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class CatDogDataset(Dataset):
    """
    Dataset class for Dog vs Cat classification.
    """

    def __init__(self, df: pd.DataFrame, transforms=None):
        self.df = df
        self.transforms = transforms

        # Determine mode based on columns
        self.has_label = "label" in df.columns
        self.has_id = "id" in df.columns

        # Pre-compute full paths to avoid doing it in __getitem__
        # Assuming 'filepath' in metadata is relative to CFG.input_dir
        self.file_paths = [
            os.path.join(CFG.input_dir, fp) for fp in df["filepath"].values
        ]

        if self.has_label:
            self.labels = df["label"].values

        if self.has_id:
            self.ids = df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Read image
        image = cv2.imread(path)
        if image is None:
            # Handle missing image gracefully (though metadata check should prevent this)
            # Return a black image or raise error. Raising error is safer for debugging.
            raise FileNotFoundError(f"Image not found at {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return logic
        if self.has_label:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        elif self.has_id:
            # For test set, return image and ID
            id_val = torch.tensor(self.ids[idx], dtype=torch.long)
            return image, id_val
        else:
            # Fallback if neither label nor id exists
            return image, torch.tensor(-1)


def make_loader(
    df: pd.DataFrame,
    image_size: int = CFG.image_size,
    batch_size: int = CFG.batch_size,
    is_train: bool = True,
):
    """
    Creates a DataLoader.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        image_size (int): Image resolution.
        batch_size (int): Batch size.
        is_train (bool): Whether this is for training (enables shuffle and train transforms).

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    transform_mode = "train" if is_train else "valid"
    transforms = get_transforms(transform_mode, image_size)

    dataset = CatDogDataset(df, transforms=transforms)

    # Generator for reproducibility
    g = torch.Generator()
    g.manual_seed(CFG.seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=is_train,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return loader
