import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from library.utils import seed_everything

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_1"

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(phase: str = "train", image_size: int = 224):
    """
    Returns the data transformation pipeline for the specified phase.

    Args:
        phase (str): 'train' for augmentation, 'val' or 'test' for deterministic resizing.
        image_size (int): The target input size for the model (default 224).

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    if phase == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        # For validation/test: Resize shorter side to 256, then center crop
        resize_dim = int(image_size * 256 / 224)
        return transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


def load_processed_metadata(load_cached_data: bool = True, cache_dir: str = CACHE_DIR):
    """
    Loads metadata and processes class labels. Implements strict caching using Parquet.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        cache_dir (str): Directory to store/load cached files.

    Returns:
        tuple: (train_df, val_df, test_df, classes_list)
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    train_cache_path = os.path.join(cache_dir, "train_meta.parquet")
    val_cache_path = os.path.join(cache_dir, "val_meta.parquet")
    test_cache_path = os.path.join(cache_dir, "test_meta.parquet")
    classes_cache_path = os.path.join(cache_dir, "classes.parquet")

    # Check if all cache files exist
    cache_exists = (
        os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
        and os.path.exists(test_cache_path)
        and os.path.exists(classes_cache_path)
    )

    if load_cached_data and cache_exists:
        print(f"Loading cached metadata from {cache_dir}...")
        train_df = pd.read_parquet(train_cache_path)
        val_df = pd.read_parquet(val_cache_path)
        test_df = pd.read_parquet(test_cache_path)
        classes_df = pd.read_parquet(classes_cache_path)
        classes = classes_df["breed"].tolist()
        return train_df, val_df, test_df, classes

    print("Processing metadata from scratch...")

    # Load raw metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Generate deterministic class mapping (sorted alphabetically)
    classes = sorted(train_df["breed"].unique().tolist())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # Map breeds to integers for training and validation sets
    train_df["label_idx"] = train_df["breed"].map(class_to_idx)
    val_df["label_idx"] = val_df["breed"].map(class_to_idx)
    # Note: test_df does not have ground truth 'breed' to map, or it is dummy.
    # We don't add label_idx to test_df.

    # Save to cache (using Parquet as requested)
    train_df.to_parquet(train_cache_path)
    val_df.to_parquet(val_cache_path)
    test_df.to_parquet(test_cache_path)

    # Save classes list as a dataframe to avoid pickle/npy issues with strings
    pd.DataFrame({"breed": classes}).to_parquet(classes_cache_path)

    return train_df, val_df, test_df, classes


class DogBreedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, mode: str = "train"):
        """
        Custom Dataset for Dog Breed Classification.

        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels (for train/val).
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'. Determines return values.
        """
        self.df = df
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths (e.g., "train/id.jpg")
        img_path = os.path.join(INPUT_DIR, row["file_path"])

        # Load image and convert to RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Fallback for safety, though data validation should prevent this
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        if self.mode == "test":
            # For test set, return image and ID to facilitate submission creation
            return image, row["id"]
        else:
            # For train/val, return image and integer label
            label = row["label_idx"]
            return image, torch.tensor(label, dtype=torch.long)
