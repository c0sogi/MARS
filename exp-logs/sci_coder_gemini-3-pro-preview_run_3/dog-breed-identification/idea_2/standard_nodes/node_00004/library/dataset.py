import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from library.config import Config
from library.utils import set_seed


def get_transforms(phase: str):
    """
    Returns the image transformations for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The composition of transforms.
    """
    mean = Config.IMAGENET_MEAN
    std = Config.IMAGENET_STD

    if phase == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(Config.IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation and Test
        return transforms.Compose(
            [
                transforms.Resize(Config.RESIZE_SIZE),
                transforms.CenterCrop(Config.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class DogDataset(Dataset):
    """
    Custom Dataset for Dog Breed Classification.
    """

    def __init__(self, df, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, file_path, and optionally label_idx).
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Flag to indicate if this is the test set (returns id instead of label).
        """
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative to input dir (e.g., "train/id.jpg")
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Fallback for missing images (though metadata validation should prevent this)
            # Create a black image
            image = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # Return image and ID for submission mapping
            return image, row["id"]
        else:
            # Return image and class index
            label = row["label_idx"]
            return image, torch.tensor(label, dtype=torch.long)


def process_data(load_cached_data=True):
    """
    Loads metadata, processes class labels, and handles caching.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, val_df, test_df, class_names)
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_meta.parquet")
    val_cache = os.path.join(cache_dir, "val_meta.parquet")
    test_cache = os.path.join(cache_dir, "test_meta.parquet")
    classes_cache = os.path.join(cache_dir, "classes.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
            and os.path.exists(classes_cache)
        ):

            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            classes_df = pd.read_parquet(classes_cache)
            class_names = classes_df["breed"].tolist()

            return train_df, val_df, test_df, class_names

    # 2. Process from scratch
    # Load raw metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Determine classes from training data
    class_names = sorted(train_df["breed"].unique().tolist())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    # Encode labels
    train_df["label_idx"] = train_df["breed"].map(class_to_idx)
    val_df["label_idx"] = val_df["breed"].map(class_to_idx)

    # Save to cache
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    classes_df = pd.DataFrame({"breed": class_names})
    classes_df.to_parquet(classes_cache, index=False)

    return train_df, val_df, test_df, class_names


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        tuple: (dataloaders_dict, class_names)
            dataloaders_dict keys: 'train', 'val', 'test'
    """
    set_seed(Config.SEED)

    # Load and process metadata
    train_df, val_df, test_df, class_names = process_data(
        load_cached_data=load_cached_data
    )

    # Handle Debug Mode
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Create Datasets
    train_dataset = DogDataset(
        train_df, transform=get_transforms("train"), is_test=False
    )

    val_dataset = DogDataset(val_df, transform=get_transforms("val"), is_test=False)

    test_dataset = DogDataset(test_df, transform=get_transforms("test"), is_test=True)

    # Create DataLoaders
    # Use drop_last=True for training to maintain batch statistics consistency
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    return dataloaders, class_names
