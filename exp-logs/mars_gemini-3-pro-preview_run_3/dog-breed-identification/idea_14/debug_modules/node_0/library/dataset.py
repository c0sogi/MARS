import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedKFold

# Import from library
from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.seed)


def get_classes(load_cached_data=True):
    """
    Returns the list of unique breed classes, sorted alphabetically.
    Caches the result to ensure consistency across folds and runs.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        list: Sorted list of breed names.
    """
    cache_path = os.path.join(Config.working_dir, "classes.parquet")

    # 1. Load Cached Data
    if load_cached_data and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        return df["breed"].tolist()

    # 2. Compute from Scratch
    # Load provided metadata
    train_df = pd.read_csv(Config.train_metadata_path)
    val_df = pd.read_csv(Config.val_metadata_path)

    # Combine to ensure we capture all classes (though stratification implies they are in both)
    combined_df = pd.concat([train_df, val_df], axis=0)

    # Get unique classes and sort alphabetically (critical for submission alignment)
    classes = sorted(combined_df["breed"].unique().tolist())

    # 3. Cache Result
    os.makedirs(Config.working_dir, exist_ok=True)
    pd.DataFrame({"breed": classes}).to_parquet(cache_path, index=False)

    return classes


def get_data_with_folds(load_cached_data=True):
    """
    Loads training and validation metadata, combines them, and assigns fold indices
    using Stratified K-Fold. Caches the result to ensure consistent splits.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        pd.DataFrame: Dataframe with 'fold' column.
    """
    cache_path = os.path.join(Config.working_dir, "folds.parquet")

    # 1. Load Cached Data
    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # 2. Compute from Scratch
    # Load metadata
    train_df = pd.read_csv(Config.train_metadata_path)
    val_df = pd.read_csv(Config.val_metadata_path)

    # Combine into a single training pool
    df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    # Create Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )
    df["fold"] = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["breed"])):
        df.loc[val_idx, "fold"] = fold

    # 3. Cache Result
    os.makedirs(Config.working_dir, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class DogDataset(Dataset):
    """
    PyTorch Dataset for Dog Breed Prediction.
    """

    def __init__(self, df, transform=None, mode="train", label2idx=None):
        self.df = df
        self.transform = transform
        self.mode = mode
        self.label2idx = label2idx

        # Prepend input directory to relative paths from metadata
        self.file_paths = [
            os.path.join(Config.input_dir, p) for p in df["file_path"].values
        ]
        self.ids = df["id"].values

        # Labels are only available in train/val modes
        if self.mode != "test":
            self.labels = df["breed"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load image using OpenCV
        img = cv2.imread(path)

        # Handle potential loading errors (though metadata validation passed)
        if img is None:
            # Return a black image of correct size to prevent crash
            img = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL for torchvision transforms
        img_pil = Image.fromarray(img)

        # Apply transforms
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = T.ToTensor()(img_pil)

        # Return dictionary
        if self.mode == "test":
            return {"image": img_tensor, "id": self.ids[idx]}
        else:
            label_str = self.labels[idx]
            label_idx = self.label2idx[label_str]
            return {
                "image": img_tensor,
                "label": torch.tensor(label_idx, dtype=torch.long),
                "id": self.ids[idx],
            }


def get_transforms(mode="train", image_size=224):
    """
    Returns the augmentation pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
        image_size (int): Input resolution.

    Returns:
        torchvision.transforms.Compose
    """
    # ImageNet statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if mode == "train":
        # Rigid preprocessing pipeline as specified
        return T.Compose(
            [
                T.RandomResizedCrop(image_size),
                T.RandomHorizontalFlip(),
                T.RandAugment(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    else:
        # Deterministic resizing for Val/Test
        # We resize to (image_size, image_size) to ensure fixed input resolution
        return T.Compose(
            [T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(mean, std)]
        )


def get_loaders(fold_idx, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold of the Cross-Validation.

    Args:
        fold_idx (int): The fold index (0-4) to use as validation.
        load_cached_data (bool): Whether to use cached fold definitions.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # 1. Prepare Data and Classes
    df = get_data_with_folds(load_cached_data=load_cached_data)
    classes = get_classes(load_cached_data=load_cached_data)
    label2idx = {c: i for i, c in enumerate(classes)}

    # 2. Split Data
    train_df = df[df["fold"] != fold_idx].copy()
    val_df = df[df["fold"] == fold_idx].copy()

    # Handle Debug Mode
    if Config.debug:
        train_df = train_df.iloc[: Config.debug_subset_size]
        val_df = val_df.iloc[: Config.debug_subset_size]

    # 3. Create Datasets
    train_dataset = DogDataset(
        train_df,
        transform=get_transforms("train", Config.image_size),
        mode="train",
        label2idx=label2idx,
    )

    val_dataset = DogDataset(
        val_df,
        transform=get_transforms("val", Config.image_size),
        mode="val",
        label2idx=label2idx,
    )

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader():
    """
    Creates DataLoader for the test set.

    Returns:
        DataLoader
    """
    # Load test metadata
    df = pd.read_csv(Config.test_metadata_path)

    if Config.debug:
        df = df.iloc[: Config.debug_subset_size]

    dataset = DogDataset(
        df, transform=get_transforms("test", Config.image_size), mode="test"
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader
