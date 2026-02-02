import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config

# Standard ImageNet normalization statistics
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class DogDataset(Dataset):
    """
    PyTorch Dataset for Dog Breed Classification.
    Handles loading images from disk and mapping labels to indices.
    """

    def __init__(self, df, class_to_idx=None, transforms=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path' and 'breed' (if not test).
            class_to_idx (dict, optional): Mapping from breed name to integer index. Required if not is_test.
            transforms (callable, optional): Transformations to apply to the image.
            is_test (bool): Flag to indicate if this is the test set (returns ID instead of label).
        """
        self.df = df
        self.transforms = transforms
        self.is_test = is_test
        self.class_to_idx = class_to_idx

        # Pre-compute full file paths
        # Metadata paths are relative to input dir, e.g., "train/id.jpg"
        self.paths = [os.path.join(Config.INPUT_DIR, p) for p in df["file_path"].values]

        if not self.is_test:
            if self.class_to_idx is None:
                raise ValueError(
                    "class_to_idx must be provided for training/validation sets."
                )
            self.labels = [self.class_to_idx[b] for b in df["breed"].values]
        else:
            self.ids = df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]

        # Load image and convert to RGB
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            # Fallback for corrupt images (though dataset is assumed clean)
            # Return a black image of the expected size
            image = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE))

        # Apply transformations
        if self.transforms:
            image = self.transforms(image)

        if self.is_test:
            # Return image and ID for submission generation
            return image, self.ids[idx]
        else:
            # Return image and label index for training
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)


def get_train_transforms(img_size=Config.IMG_SIZE):
    """
    Returns the training augmentation pipeline:
    RandomResizedCrop -> RandomHorizontalFlip -> RandAugment -> Normalize
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def get_valid_transforms(img_size=Config.IMG_SIZE):
    """
    Returns the validation/test preprocessing pipeline:
    Resize (256) -> CenterCrop (224) -> Normalize
    """
    # Standard practice: Resize to slightly larger than target crop size
    # If target is 224, resize to 256.
    resize_dim = int(img_size * 256 / 224)

    return transforms.Compose(
        [
            transforms.Resize(resize_dim),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def get_data(mode="train", load_cached_data=True):
    """
    Loads metadata and prepares the dataframe and class mappings.
    Implements caching for class mappings to ensure consistency across folds/runs.

    Args:
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to use cached class mappings if available.

    Returns:
        df (pd.DataFrame): The requested metadata dataframe.
        class_to_idx (dict): Mapping from breed name to integer index.
        idx_to_class (dict): Mapping from integer index to breed name.
    """
    # Ensure working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, "classes.parquet")

    # 1. Determine Class Mapping (Deterministic & Cached)
    classes = None

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            classes_df = pd.read_parquet(cache_path)
            classes = classes_df["breed"].tolist()
        except Exception:
            classes = None

    # If cache miss or force reload, compute from training data
    if classes is None:
        if not os.path.exists(Config.TRAIN_CSV):
            raise FileNotFoundError(
                f"Training metadata not found at {Config.TRAIN_CSV}"
            )

        train_df_full = pd.read_csv(Config.TRAIN_CSV)
        classes = sorted(train_df_full["breed"].unique().tolist())

        # Save to cache
        pd.DataFrame({"breed": classes}).to_parquet(cache_path)

    # Create mappings
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}

    # 2. Load the specific dataframe for the requested mode
    if mode == "train":
        df = pd.read_csv(Config.TRAIN_CSV)
    elif mode == "val":
        df = pd.read_csv(Config.VAL_CSV)
    elif mode == "test":
        df = pd.read_csv(Config.TEST_CSV)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'train', 'val', or 'test'.")

    # 3. Apply Debugging limits if enabled
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    return df, class_to_idx, idx_to_class
