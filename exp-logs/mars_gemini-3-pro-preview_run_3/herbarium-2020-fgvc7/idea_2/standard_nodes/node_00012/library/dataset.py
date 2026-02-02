import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from library.config import Config


def get_label_map():
    """
    Generates or loads a mapping from raw category_id to contiguous integers [0, num_classes-1].
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    map_path = os.path.join(Config.WORKING_DIR, "label_map.json")

    if os.path.exists(map_path):
        with open(map_path, "r") as f:
            # JSON keys are strings, convert back to int
            return {int(k): v for k, v in json.load(f).items()}

    print("Generating label map from training data...")
    df = pd.read_csv(Config.TRAIN_CSV)
    unique_ids = sorted(df["category_id"].unique().tolist())
    label_map = {int(raw_id): idx for idx, raw_id in enumerate(unique_ids)}

    with open(map_path, "w") as f:
        json.dump(label_map, f)

    return label_map


def get_transforms(phase: str):
    """
    Returns the data transformations for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The composed transformations.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(Config.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # For val and test
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(Config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class HerbariumDataset(Dataset):
    """
    Custom Dataset for the Herbarium 2020 competition.
    """

    def __init__(self, csv_path, transform=None, is_test=False):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Whether this is the test set (returns image_id instead of label).
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.is_test = is_test
        self.input_dir = Config.INPUT_DIR

        if not self.is_test:
            self.label_map = get_label_map()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # The file_path in csv is relative to input_dir (e.g., nybg2020/train/...)
        img_path = os.path.join(self.input_dir, row["file_path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Fallback for corrupted/missing images: return a black image
            # This prevents the dataloader from crashing
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # For test set, return image and image_id (for submission)
            image_id = row["image_id"]
            return image, image_id
        else:
            # For train/val, return image and label
            raw_label = row["category_id"]
            label = self.label_map[raw_label]
            return image, torch.tensor(label, dtype=torch.long)


def get_weighted_sampler(df, load_cached_data=True):
    """
    Calculates sample weights for Inverse Square-Root Sampling and returns a WeightedRandomSampler.
    Implements caching to disk.

    Args:
        df (pd.DataFrame): The training dataframe containing 'category_id'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        WeightedRandomSampler: The sampler.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "train_sample_weights.npy")

    sample_weights = None
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached sample weights from {cache_path}")
        loaded_weights = np.load(cache_path)
        # Cite solution_lesson_node_00010: Validate cached artifacts against runtime dimensions
        if len(loaded_weights) == len(df):
            sample_weights = loaded_weights
        else:
            print(
                f"Cached weights size ({len(loaded_weights)}) mismatch with df ({len(df)}). Recomputing..."
            )

    if sample_weights is None:
        print("Computing sample weights...")
        # Calculate class counts
        class_counts = df["category_id"].value_counts().sort_index()

        # Calculate weight per class: 1 / (count ^ power)
        # Using Config.SAMPLING_POWER (0.5 for sqrt)
        power = Config.SAMPLING_POWER
        class_weights = 1.0 / (class_counts**power)

        # Normalize weights (optional, but keeps numbers reasonable)
        class_weights = class_weights / class_weights.sum()

        # Map weights to samples
        # Create a mapping dictionary for O(1) lookup
        weight_map = class_weights.to_dict()

        # Map to the dataframe
        sample_weights = df["category_id"].map(weight_map).values.astype(np.float64)

        # Save to cache
        np.save(cache_path, sample_weights)
        print(f"Saved sample weights to {cache_path}")

    # Convert to tensor
    sample_weights_tensor = torch.from_numpy(sample_weights)

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor, num_samples=len(sample_weights), replacement=True
    )

    return sampler


def get_dataloaders(load_cached_data=True, debug_sample_size=None):
    """
    Creates and returns dataloaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached weights for the sampler.
        debug_sample_size (int, optional): If provided, limits the dataset size for debugging.

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    # 1. Load DataFrames
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debugging: subset if requested
    if debug_sample_size:
        train_df = train_df.head(debug_sample_size)
        val_df = val_df.head(debug_sample_size)
        test_df = test_df.head(debug_sample_size)

    # 2. Create Datasets
    train_dataset = HerbariumDataset(
        (
            Config.TRAIN_CSV if not debug_sample_size else Config.TRAIN_CSV
        ),  # Re-reading inside class, but passing path is cleaner for API
        transform=get_transforms("train"),
        is_test=False,
    )
    # Patch dataset df if debugging (since we passed path to init)
    if debug_sample_size:
        train_dataset.df = train_df

    val_dataset = HerbariumDataset(
        Config.VAL_CSV, transform=get_transforms("val"), is_test=False
    )
    if debug_sample_size:
        val_dataset.df = val_df

    test_dataset = HerbariumDataset(
        Config.TEST_CSV, transform=get_transforms("test"), is_test=True
    )
    if debug_sample_size:
        test_dataset.df = test_df

    # 3. Create Sampler for Training
    # We use the dataframe we loaded (potentially subsetted) to calculate weights
    train_sampler = get_weighted_sampler(train_df, load_cached_data=load_cached_data)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
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

    return {"train": train_loader, "val": val_loader, "test": test_loader}
