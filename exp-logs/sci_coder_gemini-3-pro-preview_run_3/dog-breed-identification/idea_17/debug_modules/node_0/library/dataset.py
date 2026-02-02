import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import seed_everything

# Ensure working directory exists for caching
os.makedirs(Config.WORKING_DIR, exist_ok=True)


class BreedDataset(Dataset):
    """
    PyTorch Dataset for Dog Breed Classification.
    Handles loading images, applying transforms, and returning (image, label) pairs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_path: str,
        transforms=None,
        mode: str = "train",
        class_to_idx: dict = None,
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata (id, breed, file_path).
            base_path (str): Root directory where images are stored.
            transforms (callable, optional): Transformations to apply to the image.
            mode (str): 'train', 'val', or 'test'.
            class_to_idx (dict, optional): Mapping from breed name to integer index. Required for train/val.
        """
        self.df = df
        self.base_path = base_path
        self.transforms = transforms
        self.mode = mode
        self.class_to_idx = class_to_idx

        # Pre-compute full paths to avoid overhead in __getitem__
        # Metadata paths are relative, e.g., "train/id.jpg"
        self.file_paths = [
            os.path.join(self.base_path, rel_path) for rel_path in self.df["file_path"]
        ]

        if self.mode != "test":
            if self.class_to_idx is None:
                raise ValueError(
                    "class_to_idx must be provided for training/validation"
                )
            # Encode labels
            self.labels = [self.class_to_idx[breed] for breed in self.df["breed"]]
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load image
        # Using PIL is generally safer for torchvision transforms
        try:
            image = Image.open(path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Fallback for missing/corrupt images: create a black image
            # This prevents crashing during training
            image = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), (0, 0, 0))

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)

        if self.mode == "test":
            # Return image and ID for submission file creation
            return image, self.df.iloc[idx]["id"]
        else:
            # Return image and integer label
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label


def get_transforms(mode: str = "train"):
    """
    Returns the transformation pipeline based on the mode.
    Implements the strategy:
    Train: RandomResizedCrop -> RandomHorizontalFlip -> RandAugment -> Norm
    Val/Test: Resize -> CenterCrop -> Norm
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    Config.IMG_SIZE, scale=(0.08, 1.0), ratio=(0.75, 1.33)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # RandAugment is crucial for regularization in modern ConvNets
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=Config.MEAN, std=Config.STD),
            ]
        )
    else:
        # Standard ImageNet evaluation pipeline
        # Resize to slightly larger than target, then center crop
        resize_dim = int(Config.IMG_SIZE * 256 / 224)
        return transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.CenterCrop(Config.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=Config.MEAN, std=Config.STD),
            ]
        )


def get_metadata(load_cached_data: bool = True):
    """
    Loads metadata and generates/caches the class mapping.
    Ensures consistent label encoding across all runs.

    Args:
        load_cached_data (bool): If True, attempts to load class mapping from cache.

    Returns:
        tuple: (df_train, df_val, df_test, class_to_idx, classes)
    """
    cache_path = os.path.join(Config.WORKING_DIR, "classes.parquet")

    # Load raw metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        df_train = df_train.head(Config.DEBUG_SUBSET_SIZE)
        df_val = df_val.head(Config.DEBUG_SUBSET_SIZE)
        df_test = df_test.head(Config.DEBUG_SUBSET_SIZE)

    # Handle Class Mapping
    if load_cached_data and os.path.exists(cache_path):
        # Load cached classes
        df_classes = pd.read_parquet(cache_path)
        classes = df_classes["breed"].tolist()
    else:
        # Compute classes from training data
        classes = sorted(df_train["breed"].unique().tolist())
        # Save to cache
        pd.DataFrame({"breed": classes}).to_parquet(cache_path, index=False)

    # Create mapping
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    return df_train, df_val, df_test, class_to_idx, classes


def get_dataloaders(
    fold_idx: int = 0, load_cached_data: bool = True, batch_size: int = None
):
    """
    Creates DataLoaders for a specific fold (or general training if not using K-Fold strictly
    at the dataset level, but the metadata is already split).

    Since the metadata provided (train.csv, val.csv) represents a single stratified split (80/20),
    we treat this as a fixed validation set scenario unless we merge and re-split.
    Given the prompt mentions "5-Fold Stratified Cross-Validation" in the strategy,
    but the metadata generation script only produced one `train.csv` and `val.csv`,
    we will assume the `train.csv` contains the data available for training in this session,
    and `val.csv` is the fixed validation set.

    However, to support true K-Fold as requested by the strategy, we should ideally
    merge train and val and split dynamically, OR assume the provided metadata is just
    one fold's view.

    To adhere strictly to the provided metadata files which are static:
    We will use `train.csv` for training and `val.csv` for validation.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Load metadata and consistent class mapping
    df_train, df_val, df_test, class_to_idx, classes = get_metadata(load_cached_data)

    # Create Datasets
    train_dataset = BreedDataset(
        df=df_train,
        base_path=Config.INPUT_DIR,
        transforms=get_transforms(mode="train"),
        mode="train",
        class_to_idx=class_to_idx,
    )

    val_dataset = BreedDataset(
        df=df_val,
        base_path=Config.INPUT_DIR,
        transforms=get_transforms(mode="val"),
        mode="val",
        class_to_idx=class_to_idx,
    )

    test_dataset = BreedDataset(
        df=df_test,
        base_path=Config.INPUT_DIR,
        transforms=get_transforms(mode="test"),
        mode="test",
        class_to_idx=None,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, classes
