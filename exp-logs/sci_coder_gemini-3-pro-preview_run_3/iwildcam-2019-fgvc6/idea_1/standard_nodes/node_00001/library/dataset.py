import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config


def get_transforms(split="train"):
    """
    Returns albumentations transforms for train/val/test splits.
    """
    mean = Config.MEAN
    std = Config.STD
    img_size = Config.IMG_SIZE

    if split == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class AnimalDataset(Dataset):
    def __init__(self, df, transforms=None, is_test=False):
        self.df = df
        self.transforms = transforms
        self.is_test = is_test
        self.input_root = Config.INPUT_ROOT

        # Pre-compute paths to avoid overhead in __getitem__
        self.file_paths = df["file_path"].values
        self.ids = df["Id"].values

        if not self.is_test:
            self.labels = df["Category"].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full path
        path = os.path.join(self.input_root, self.file_paths[idx])

        # Read image
        image = cv2.imread(path)
        if image is None:
            # Handle missing image/read error by returning a black image
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        image_id = self.ids[idx]

        if self.is_test:
            return image, image_id
        else:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long), image_id


def load_and_process_metadata(load_cached_data=True):
    """
    Loads metadata CSVs, handles debug subsetting, and caches the result to parquet.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache paths
    suffix = "_debug" if Config.DEBUG else ""
    train_cache = os.path.join(cache_dir, f"train_processed{suffix}.parquet")
    val_cache = os.path.join(cache_dir, f"val_processed{suffix}.parquet")
    test_cache = os.path.join(cache_dir, f"test_processed{suffix}.parquet")

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception:
                # If load fails, proceed to compute from scratch
                pass

    # 2. Compute/Process from scratch
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Handle Debug Mode
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

    # 3. Save to cache
    train_df.to_parquet(train_cache)
    val_df.to_parquet(val_cache)
    test_df.to_parquet(test_cache)

    return train_df, val_df, test_df


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.
    Implements WeightedRandomSampler for the training set to handle class imbalance.
    """
    train_df, val_df, test_df = load_and_process_metadata(
        load_cached_data=load_cached_data
    )

    # --- Weighted Random Sampler for Training ---
    # Calculate class weights based on inverse frequency
    class_counts = train_df["Category"].value_counts().sort_index()
    count_map = class_counts.to_dict()

    # Assign a weight to each sample corresponding to its class
    sample_weights = []
    for category in train_df["Category"]:
        count = count_map.get(category, 0)
        if count > 0:
            sample_weights.append(1.0 / count)
        else:
            sample_weights.append(0.0)

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # --- Datasets ---
    train_dataset = AnimalDataset(
        train_df, transforms=get_transforms("train"), is_test=False
    )

    val_dataset = AnimalDataset(val_df, transforms=get_transforms("val"), is_test=False)

    test_dataset = AnimalDataset(
        test_df, transforms=get_transforms("test"), is_test=True
    )

    # --- DataLoaders ---
    # Train loader uses sampler, so shuffle must be False
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
