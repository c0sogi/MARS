import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def get_transforms(data="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Valid and Test
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transforms=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            transforms (A.Compose): Albumentations transforms.
        """
        self.df = df
        self.transforms = transforms
        self.target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

        # Check if targets exist in dataframe (they won't for test set)
        self.has_labels = all(col in df.columns for col in self.target_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths like "images/Train_0.jpg"
        # Config.INPUT_DIR is "./input"
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transforms provided
            t = ToTensorV2()
            image = t(image=image)["image"]

        # Handle Labels
        if self.has_labels:
            labels = row[self.target_cols].values.astype(np.float32)
            return image, torch.tensor(labels)
        else:
            # Return dummy labels for test set to maintain signature consistency
            return image, torch.zeros(len(self.target_cols), dtype=torch.float32)


def generate_stratified_bags(df, num_bags, seed=42, load_cached_data=True):
    """
    Generates stratified bags (bootstrap samples) and OOB validation sets.
    Caches the indices to a parquet file.

    Args:
        df (pd.DataFrame): The full dataset.
        num_bags (int): Number of bags to generate.
        seed (int): Random seed.
        load_cached_data (bool): Whether to load from cache.

    Returns:
        list: A list of tuples (train_indices, oob_indices).
    """
    cache_path = os.path.join(Config.CACHE_DIR, "bags.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            bags = []
            for _, row in cache_df.iterrows():
                bags.append((row["train_idx"], row["oob_idx"]))
            # print(f"Loaded {len(bags)} bags from cache.")
            return bags
        except Exception as e:
            # print(f"Failed to load cache: {e}. Recomputing.")
            pass

    # 2. Compute from scratch
    # Ensure stratify_label exists
    if "stratify_label" not in df.columns:
        # Reconstruct stratify label based on max value of targets
        target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
        df["stratify_label"] = df[target_cols].idxmax(axis=1)

    rng = np.random.RandomState(seed)
    bags = []
    cache_data = []

    all_indices = df.index.values

    for i in range(num_bags):
        bag_train_indices = []

        # Stratified sampling with replacement
        for label, group in df.groupby("stratify_label"):
            group_indices = group.index.values
            # Sample with replacement, size = len(group)
            sampled = rng.choice(group_indices, size=len(group_indices), replace=True)
            bag_train_indices.extend(sampled)

        bag_train_indices = np.array(bag_train_indices)

        # OOB indices (Validation)
        # Set difference: All indices - Unique indices in bag
        unique_bag_indices = np.unique(bag_train_indices)
        bag_oob_indices = np.setdiff1d(all_indices, unique_bag_indices)

        bags.append((bag_train_indices, bag_oob_indices))

        cache_data.append(
            {"bag_id": i, "train_idx": bag_train_indices, "oob_idx": bag_oob_indices}
        )

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_df = pd.DataFrame(cache_data)
    cache_df.to_parquet(cache_path)

    return bags


def get_bag_loaders(bag_idx):
    """
    Creates DataLoaders for a specific bag index.
    Combines train and val metadata to form the full pool, then splits based on the bag.

    Args:
        bag_idx (int): The index of the bag (0 to NUM_BAGS-1).

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load and combine metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Concatenate to form full dataset
    # ignore_index=True is crucial so we have a continuous index 0..N
    full_df = pd.concat([train_meta, val_meta], ignore_index=True)

    # Generate bags
    bags = generate_stratified_bags(
        full_df, num_bags=Config.NUM_BAGS, seed=Config.SEED, load_cached_data=True
    )

    if bag_idx >= len(bags):
        raise ValueError(f"Bag index {bag_idx} out of range (Total bags: {len(bags)})")

    train_idx, oob_idx = bags[bag_idx]

    # Create DataFrames
    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[oob_idx].reset_index(drop=True)

    # Create Datasets
    train_dataset = AppleDataset(train_df, transforms=get_transforms("train"))
    val_dataset = AppleDataset(val_df, transforms=get_transforms("valid"))

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader():
    """
    Creates DataLoader for the test set.

    Returns:
        DataLoader: Test data loader.
    """
    df = pd.read_csv(Config.TEST_METADATA_PATH)

    dataset = AppleDataset(df, transforms=get_transforms("test"))

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
