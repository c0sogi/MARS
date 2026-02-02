import os
import cv2
import torch
import pandas as pd

# Disable OpenCV threading to prevent DataLoader deadlocks
cv2.setNumThreads(0)
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config
from library.utils import set_seed


class PlantDataset(Dataset):
    """
    Custom Dataset for Herbarium 2020 Plant Species Classification.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels/ids.
            transforms (albumentations.Compose): Transformations to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-compute full paths to avoid overhead in __getitem__
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, path) for path in df["file_path"].values
        ]

        if self.mode != "test":
            self.labels = df["category_id"].values
        else:
            self.image_ids = df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Read image using OpenCV
        image = cv2.imread(path)
        if image is None:
            # Fallback for missing/corrupt images (though analysis showed none)
            # Create a black image to prevent crashing
            image = np.zeros(
                (Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.mode != "test":
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            image_id = self.image_ids[idx]
            return image, torch.tensor(image_id, dtype=torch.long)


def get_transforms(split="train"):
    """
    Returns the Albumentations transformations for the specified split.
    """
    mean = Config.MEAN
    std = Config.STD
    height, width = Config.IMG_SIZE

    if split == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(size=(height, width), scale=(0.08, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Resize short edge to 256, then CenterCrop 224
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=256),
                A.CenterCrop(height=height, width=width),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_sampler_weights(df, load_cached_data=True):
    """
    Calculates or loads sample weights for WeightedRandomSampler.

    Args:
        df (pd.DataFrame): Training dataframe containing 'category_id'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Array of weights corresponding to each sample in df.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Create a cache filename based on dataset size to avoid conflicts (e.g. debug vs full)
    cache_path = os.path.join(Config.WORKING_DIR, f"train_sample_weights_{len(df)}.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            if len(weights) == len(df):
                # print(f"Loaded sampler weights from {cache_path}") # Suppressed as per instructions
                return weights
        except Exception:
            pass  # Fallback to recomputing

    # Compute weights
    # 1. Count samples per class
    class_counts = df["category_id"].value_counts().sort_index()

    # 2. Compute weight per class (1 / count)
    # Ensure we cover all classes up to NUM_CLASSES, though df might be a subset
    # We map actual category_ids present in df.

    # Create a mapping from category_id to weight
    class_weights = 1.0 / class_counts

    # 3. Map weights to samples
    # Using map is faster than iterating
    sample_weights = df["category_id"].map(class_weights).values.astype(np.float64)

    # Save to cache
    np.save(cache_path, sample_weights)

    return sample_weights


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached artifacts.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed(Config.SEED)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debugging: Subsample if configured
    if Config.DEBUG_SAMPLE_SIZE is not None:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)
        # Disable loading cached weights if we are debugging to ensure weights match subset
        load_cached_data = False

    # Prepare Transforms
    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    # Create Datasets
    train_dataset = PlantDataset(train_df, transforms=train_transforms, mode="train")
    val_dataset = PlantDataset(val_df, transforms=val_transforms, mode="val")
    test_dataset = PlantDataset(test_df, transforms=val_transforms, mode="test")

    # Prepare WeightedRandomSampler for Training
    sample_weights = get_sampler_weights(train_df, load_cached_data=load_cached_data)
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Sampler is mutually exclusive with shuffle
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
