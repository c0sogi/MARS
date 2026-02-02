import os
import json
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("data_module")


class HerbariumDataset(Dataset):
    """
    Custom Dataset for Herbarium 2020 FGVC7.
    Reads images via OpenCV and applies Albumentations transforms.
    """

    def __init__(self, df, root_dir, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            root_dir (str): Root directory for image paths.
            transform (albumentations.Compose): Transformations to apply.
            is_test (bool): If True, returns dummy target.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path
        file_path = os.path.join(self.root_dir, row["file_path"])

        # Read image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing/corrupt images: return a black image
            # This prevents crashing during training
            image = np.zeros((300, 300, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided
            base_transform = A.Compose(
                [A.Resize(300, 300), A.Normalize(), ToTensorV2()]
            )
            augmented = base_transform(image=image)
            image = augmented["image"]

        # Return data
        if self.is_test:
            # For test, we need image_id for submission
            return image, row["image_id"]
        else:
            # For train/val, we return category_id
            return image, torch.tensor(row["category_id"], dtype=torch.long)


def get_taxonomy_mapping(load_cached_data=True):
    """
    Generates or loads a mapping from category_id (species) to genus_id.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: DataFrame with columns ['category_id', 'genus_id', 'genus_name'].
    """
    cache_path = Config.TAXONOMY_MAP_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading taxonomy mapping from cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Regenerating...")

    # 2. Regenerate mapping
    logger.info("Generating taxonomy mapping from raw metadata...")

    # Load raw metadata (this is large, ~10M lines, so it takes a moment)
    raw_meta_path = os.path.join(Config.INPUT_DIR, "nybg2020/train/metadata.json")

    with open(raw_meta_path, "r") as f:
        data = json.load(f)

    categories = data["categories"]
    cat_df = pd.DataFrame(categories)

    # We are interested in 'id' (category_id) and 'genus'
    # Create a unique ID for each genus
    cat_df["genus_id"] = pd.factorize(cat_df["genus"])[0]

    result_df = cat_df[["id", "genus_id", "genus"]].rename(
        columns={"id": "category_id", "genus": "genus_name"}
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Save to cache
    result_df.to_parquet(cache_path, index=False)
    logger.info(f"Taxonomy mapping saved to {cache_path}")

    return result_df


def get_transforms(img_size, phase="train"):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        img_size (int): Target image size (height and width).
        phase (str): 'train' or 'val'/'test'.
    """
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_dataloaders(img_size, batch_size, debug=False):
    """
    Creates DataLoaders for training and validation.

    Args:
        img_size (int): Image resolution.
        batch_size (int): Batch size.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        train_loader, val_loader
    """
    logger.info(
        f"Preparing dataloaders: size={img_size}, batch={batch_size}, debug={debug}"
    )

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    if debug:
        subset_size = Config.DEBUG_SUBSET_SIZE
        logger.info(f"Debug mode: Subsetting to {subset_size} samples.")
        train_df = train_df.sample(
            n=min(len(train_df), subset_size), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), subset_size), random_state=Config.SEED
        ).reset_index(drop=True)

    # Define Transforms
    train_transform = get_transforms(img_size, phase="train")
    val_transform = get_transforms(img_size, phase="val")

    # Create Datasets
    train_dataset = HerbariumDataset(
        train_df, Config.INPUT_DIR, transform=train_transform, is_test=False
    )
    val_dataset = HerbariumDataset(
        val_df, Config.INPUT_DIR, transform=val_transform, is_test=False
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
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(img_size, batch_size):
    """
    Creates DataLoader for the test set.
    """
    logger.info(f"Preparing test dataloader: size={img_size}, batch={batch_size}")

    test_df = pd.read_csv(Config.TEST_CSV)

    test_transform = get_transforms(img_size, phase="test")

    test_dataset = HerbariumDataset(
        test_df, Config.INPUT_DIR, transform=test_transform, is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
