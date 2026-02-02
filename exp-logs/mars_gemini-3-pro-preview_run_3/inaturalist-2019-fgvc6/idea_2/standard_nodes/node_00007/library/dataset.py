import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from library import config
from library import utils

# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class INatDataset(Dataset):
    """
    PyTorch Dataset for iNaturalist 2019.
    Reads images based on metadata paths and applies transforms.
    """

    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_name', 'image_id', and optionally 'category_id'.
            root_dir (str): Root directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.root_dir, row["file_name"])

        # Load image using OpenCV
        img = cv2.imread(file_path)

        # Handle missing or corrupt images gracefully
        if img is None:
            # Create a blank image (black) of standard size to prevent crashing
            # In a real scenario, logging this would be good, but we keep it silent as per instructions
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for compatibility with timm/torchvision transforms
        img = Image.fromarray(img)

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Get target if available (Training/Validation), else -1 (Test)
        target = row["category_id"] if "category_id" in row else -1
        image_id = row["image_id"]

        return img, target, image_id


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------


def get_transforms(image_size, is_training, rand_augment=False, ra_params=None):
    """
    Creates image transformations using timm.

    Args:
        image_size (int): Target image size (e.g., 224, 384).
        is_training (bool): Whether to generate training transforms (augmentation) or validation.
        rand_augment (bool): Whether to apply RandAugment during training.
        ra_params (dict): Dictionary with 'magnitude' and 'num_ops' for RandAugment.

    Returns:
        torchvision.transforms.Compose: Composed transforms.
    """
    auto_augment = None
    if is_training and rand_augment and ra_params:
        # Construct timm auto_augment string: rand-m{magnitude}-n{num_ops}-mstd0.5
        # mstd0.5 matches the "std dev 0.5" requirement in the idea
        magnitude = ra_params.get("magnitude", 9)
        num_ops = ra_params.get("num_ops", 2)
        auto_augment = f"rand-m{magnitude}-n{num_ops}-mstd0.5"

    transform = create_transform(
        input_size=(3, image_size, image_size),
        is_training=is_training,
        use_prefetcher=False,
        no_aug=not is_training,
        scale=None,  # Use default scale for RandomResizedCrop
        ratio=None,  # Use default ratio
        hflip=0.5,  # Horizontal flip probability
        vflip=0.0,
        color_jitter=(
            0.4 if is_training and not auto_augment else None
        ),  # Use jitter if not using AutoAugment
        auto_augment=auto_augment,
        interpolation="bicubic",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )

    return transform


# -----------------------------------------------------------------------------
# Data Loading Helper
# -----------------------------------------------------------------------------


def load_metadata(debug_size=None):
    """
    Loads train, validation, and test metadata from CSV files.

    Args:
        debug_size (int, optional): If provided, samples this many rows from each dataframe.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    if debug_size is not None:
        train_df = train_df.head(debug_size)
        val_df = val_df.head(debug_size)
        test_df = test_df.head(debug_size)

    return train_df, val_df, test_df


# -----------------------------------------------------------------------------
# DataLoader Factory
# -----------------------------------------------------------------------------


def get_dataloader(
    df,
    image_size,
    batch_size,
    is_training,
    sampling_strategy="instance_balanced",
    rand_augment_config=None,
):
    """
    Creates a DataLoader for the given dataframe.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        image_size (int): Input image resolution.
        batch_size (int): Batch size.
        is_training (bool): True for training set, False for val/test.
        sampling_strategy (str): 'instance_balanced' or 'class_balanced'.
        rand_augment_config (dict, optional): Configuration for RandAugment (e.g., {'magnitude': 9, 'num_ops': 2}).

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # 1. Prepare Transforms
    # Extract RA params if provided
    ra_params = {}
    use_ra = False
    if is_training and rand_augment_config:
        use_ra = rand_augment_config.get("rand_augment", False)
        if use_ra:
            ra_params = {
                "magnitude": rand_augment_config.get("rand_augment_magnitude", 9),
                "num_ops": rand_augment_config.get("rand_augment_num_ops", 2),
            }

    transform = get_transforms(
        image_size=image_size,
        is_training=is_training,
        rand_augment=use_ra,
        ra_params=ra_params,
    )

    # 2. Create Dataset
    dataset = INatDataset(df, root_dir=config.INPUT_DIR, transform=transform)

    # 3. Configure Sampler
    sampler = None
    shuffle = is_training  # Default shuffle for training if no sampler is used

    if is_training and sampling_strategy == "class_balanced":
        # Calculate weights for WeightedRandomSampler
        # Weight = 1 / frequency
        if "category_id" in df.columns:
            class_counts = df["category_id"].value_counts().sort_index()
            # Create a map from category_id to weight
            weight_map = {cat: 1.0 / count for cat, count in class_counts.items()}
            # Map weights to each sample
            sample_weights = df["category_id"].map(weight_map).fillna(0).values

            # Create sampler
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(df),
                replacement=True,
            )
            shuffle = False  # Shuffle is mutually exclusive with sampler
        else:
            # Fallback if category_id is missing (should not happen in training)
            shuffle = True

    elif is_training and sampling_strategy == "instance_balanced":
        shuffle = True
        sampler = None

    else:
        # Validation / Test
        shuffle = False
        sampler = None

    # 4. Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=is_training,  # Drop last batch during training to maintain batch statistics
    )

    return loader
