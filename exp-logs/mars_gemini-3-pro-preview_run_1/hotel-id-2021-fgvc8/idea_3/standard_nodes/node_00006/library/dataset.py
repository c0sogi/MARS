import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config


def get_transforms(stage="train"):
    """
    Returns the transformation pipeline for the specified stage.

    Args:
        stage (str): 'train', 'valid', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if stage == "train":
        # TrivialAugmentWide for diverse augmentation on the long-tail dataset
        return transforms.Compose(
            [
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    elif stage == "valid" or stage == "test":
        # Deterministic transforms for evaluation
        return transforms.Compose(
            [
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")


def process_data(load_cached_data=True, debug=Config.DEBUG):
    """
    Loads metadata, performs label encoding, and caches the result.
    Strictly follows the caching logic requirement using Parquet and NPY.

    Args:
        load_cached_data (bool): Whether to try loading from parquet cache.
        debug (bool): If True, subsets the data for debugging.

    Returns:
        tuple: (train_df, val_df, test_df, class_map)
    """
    # Define cache paths
    train_parquet = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    val_parquet = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    test_parquet = os.path.join(Config.WORKING_DIR, "test_processed.parquet")
    classes_npy = os.path.join(Config.WORKING_DIR, "classes.npy")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(train_parquet)
            and os.path.exists(val_parquet)
            and os.path.exists(test_parquet)
            and os.path.exists(classes_npy)
        ):

            train_df = pd.read_parquet(train_parquet)
            val_df = pd.read_parquet(val_parquet)
            test_df = pd.read_parquet(test_parquet)
            class_map = np.load(classes_npy, allow_pickle=True).item()

            # Apply debug subsetting if requested (after loading cache)
            if debug:
                train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
                val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
                test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

            return train_df, val_df, test_df, class_map

    # 2. Process from scratch

    # Load raw metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Generate Label Mapping
    # Ensure consistent mapping by sorting unique IDs
    unique_hotels = sorted(train_df["hotel_id"].unique())
    class_map = {hotel_id: i for i, hotel_id in enumerate(unique_hotels)}

    # Apply mapping
    # We use .map() and fillna(-1) for safety.
    # Note: val_df is a subset of train classes, so no missing keys expected.
    train_df["label"] = train_df["hotel_id"].map(class_map).fillna(-1).astype(int)
    val_df["label"] = val_df["hotel_id"].map(class_map).fillna(-1).astype(int)

    # Test set has no valid labels for training (placeholder -1)
    test_df["label"] = -1

    # Save to cache (Full datasets)
    train_df.to_parquet(train_parquet)
    val_df.to_parquet(val_parquet)
    test_df.to_parquet(test_parquet)
    np.save(classes_npy, class_map)

    # Apply debug subsetting if requested
    if debug:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    return train_df, val_df, test_df, class_map


class HotelDataset(Dataset):
    """
    PyTorch Dataset for Hotel Identification.
    Reads images from disk and applies transformations.
    """

    def __init__(self, df, transform=None, data_root=Config.INPUT_DIR):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'label'.
            transform (callable, optional): Transform pipeline.
            data_root (str): Root directory for images.
        """
        self.df = df
        self.transform = transform
        self.data_root = data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # file_path in metadata is relative to INPUT_DIR
        file_path = row["file_path"]
        full_path = os.path.join(self.data_root, file_path)

        # Load image
        # Using cv2 for robust loading
        image = cv2.imread(full_path)

        if image is None:
            # Handle missing/corrupt images by returning a blank image
            # This prevents the training loop from crashing
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms
        image = Image.fromarray(image)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get label
        label = row["label"]

        return image, torch.tensor(label, dtype=torch.long)
