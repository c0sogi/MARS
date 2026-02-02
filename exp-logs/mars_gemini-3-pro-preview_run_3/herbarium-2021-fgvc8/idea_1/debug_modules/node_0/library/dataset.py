import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config


class HerbariumDataset(Dataset):
    """
    Custom Dataset for Herbarium Plant Species Classification.
    """

    def __init__(self, df, transform=None, class_mapping=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_path, image_id, category_id).
            transform (albumentations.Compose): Transformations to apply to the image.
            class_mapping (dict): Mapping from category_id to class_idx. Required for train/val.
            is_test (bool): Flag to indicate if this is the test set.
        """
        self.df = df
        self.transform = transform
        self.class_mapping = class_mapping
        self.is_test = is_test

        # Pre-compute paths to avoid overhead in __getitem__
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, path) for path in df["file_path"].values
        ]

        if not self.is_test:
            # For train/val, we map category_id to class_idx
            self.category_ids = df["category_id"].values
            self.labels = [self.class_mapping[cat_id] for cat_id in self.category_ids]
        else:
            # For test, we keep track of image_id for submission
            self.image_ids = df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load image using OpenCV
        image = cv2.imread(path)
        if image is None:
            # Fallback or error handling; usually datasets are clean per verification
            raise FileNotFoundError(f"Image not found at {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.is_test:
            # Return image and image_id for submission mapping
            return image, self.image_ids[idx]
        else:
            # Return image and mapped class index
            return image, torch.tensor(self.labels[idx], dtype=torch.long)


def get_class_mapping(train_df, load_cached_data=True):
    """
    Generates or loads a mapping between raw category_id and model class_idx.

    Args:
        train_df (pd.DataFrame): Training metadata.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Mapping {category_id: class_idx}
    """
    cache_path = Config.CLASS_MAPPING_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            mapping_df = pd.read_parquet(cache_path)
            mapping = dict(zip(mapping_df["category_id"], mapping_df["class_idx"]))
            return mapping
        except Exception as e:
            print(f"Failed to load class mapping from cache: {e}. Recomputing...")

    # 2. Compute from scratch
    unique_categories = sorted(train_df["category_id"].unique())
    mapping = {cat_id: idx for idx, cat_id in enumerate(unique_categories)}

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    mapping_df = pd.DataFrame(
        {"category_id": list(mapping.keys()), "class_idx": list(mapping.values())}
    )
    mapping_df.to_parquet(cache_path, index=False)

    return mapping


def get_transforms():
    """
    Defines Albumentations transforms for Train and Validation/Test.
    """
    # Normalization stats from Config
    mean = Config.MEAN
    std = Config.STD

    # Train: Resize -> RandomResizedCrop -> Flip -> Normalize
    # Note: We use RandomResizedCrop as the primary augmentation.
    # Config says "All images are resized to a standard resolution... slight Random Resized Crop".
    # To be safe and efficient, we can use RandomResizedCrop directly on the original image
    # to target size 224x224.
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(
                height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE, scale=(0.8, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    # Val/Test: Resize -> Normalize
    val_transform = A.Compose(
        [
            A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform


def get_dataloaders(
    train_csv_path=Config.TRAIN_CSV,
    val_csv_path=Config.VAL_CSV,
    test_csv_path=Config.TEST_CSV,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    load_cached_data=True,
):
    """
    Constructs DataLoaders for Train, Validation, and Test sets.

    Args:
        train_csv_path, val_csv_path, test_csv_path: Paths to metadata CSVs.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of worker processes.
        debug: If True, subsets data for quick debugging.
        load_cached_data: Whether to use cached artifacts (class mapping).

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader, 'num_classes': int}
    """
    # Load Metadata
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Debug Subsampling
    if debug:
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        # Keep test full or sample? Usually debug implies checking training loop, but let's sample test too to be safe
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Get Class Mapping
    class_mapping = get_class_mapping(train_df, load_cached_data=load_cached_data)
    num_classes = len(class_mapping)

    # Prepare Transforms
    train_transform, val_transform = get_transforms()

    # Calculate Weights for WeightedRandomSampler (Train only)
    # 1. Count frequency of each class in the current training set
    class_counts = train_df["category_id"].value_counts().to_dict()

    # 2. Compute weight for each class (1 / frequency)
    # We use raw category_id for lookup
    class_weights = {cat_id: 1.0 / count for cat_id, count in class_counts.items()}

    # 3. Assign weight to each sample
    sample_weights = [class_weights[cat_id] for cat_id in train_df["category_id"]]

    # 4. Create Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_df), replacement=True
    )

    # Instantiate Datasets
    train_dataset = HerbariumDataset(
        train_df, transform=train_transform, class_mapping=class_mapping, is_test=False
    )

    val_dataset = HerbariumDataset(
        val_df, transform=val_transform, class_mapping=class_mapping, is_test=False
    )

    test_dataset = HerbariumDataset(
        test_df, transform=val_transform, class_mapping=None, is_test=True
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler instead of shuffle=True
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

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "num_classes": num_classes,
        "class_mapping": class_mapping,
    }
