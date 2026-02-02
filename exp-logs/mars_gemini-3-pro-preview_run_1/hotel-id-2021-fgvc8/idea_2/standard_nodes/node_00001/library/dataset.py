import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from library.utils import Config


def get_transforms(phase="train"):
    """
    Returns the image transformations for the specified phase.
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Base transforms: Resize to 256, then CenterCrop to 224
    common_transforms = [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((Config.IMG_SIZE, Config.IMG_SIZE)),
    ]

    if phase == "train":
        # Add Horizontal Flip for training
        return transforms.Compose(
            [
                *common_transforms,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        # Validation / Test
        return transforms.Compose(
            [
                *common_transforms,
                transforms.ToTensor(),
                normalize,
            ]
        )


class HotelDataset(Dataset):
    """
    PyTorch Dataset for Hotel Identification.
    """

    def __init__(self, df, transform=None, input_dir=Config.INPUT_DIR, is_test=False):
        self.df = df
        self.transform = transform
        self.input_dir = input_dir
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct file path
        # Metadata file_path is relative to input_dir
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (though EDA showed none)
            # Create a black image
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, row["image"]
        else:
            # Return image and integer label
            label = torch.tensor(row["label_idx"], dtype=torch.long)
            return image, label


def process_metadata(load_cached_data=False):
    """
    Loads metadata, encodes labels, and handles caching.

    Returns:
        train_df, val_df, test_df, encoder_classes
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")
    classes_cache = os.path.join(cache_dir, "classes.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(classes_cache)
    )

    if load_cached_data and cache_exists:
        print("Loading cached metadata...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        encoder_classes = np.load(classes_cache, allow_pickle=True)
        return train_df, val_df, test_df, encoder_classes

    print("Processing metadata from scratch...")

    # Load raw metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Initialize and fit LabelEncoder
    # We fit on train_df['hotel_id'] as it contains all classes (including singletons)
    encoder = LabelEncoder()
    train_df["label_idx"] = encoder.fit_transform(train_df["hotel_id"])

    # Transform validation set
    # Note: val classes are a subset of train classes
    val_df["label_idx"] = encoder.transform(val_df["hotel_id"])

    # Save to cache
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)
    np.save(classes_cache, encoder.classes_)

    return train_df, val_df, test_df, encoder.classes_


def get_dataloaders(
    load_cached_data=False, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to load processed dataframes from cache.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker processes.

    Returns:
        train_loader, val_loader, test_loader, encoder_classes
    """
    # Process/Load Metadata
    train_df, val_df, test_df, encoder_classes = process_metadata(load_cached_data)

    # Create Datasets
    train_dataset = HotelDataset(
        df=train_df, transform=get_transforms("train"), is_test=False
    )

    val_dataset = HotelDataset(
        df=val_df, transform=get_transforms("val"), is_test=False
    )

    test_dataset = HotelDataset(
        df=test_df, transform=get_transforms("test"), is_test=True
    )

    # Create DataLoaders
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, encoder_classes
