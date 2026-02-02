import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_dataset_metadata(split_name, load_cached_data=True):
    """
    Loads metadata for a given split (train, val, test).
    Implements caching using Parquet as per requirements.
    """
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)
    cache_path = os.path.join(Config.working_dir, f"cached_{split_name}.parquet")

    # Define source path based on split
    if split_name == "train":
        source_path = Config.train_csv
    elif split_name == "val":
        source_path = Config.val_csv
    elif split_name == "test":
        source_path = Config.test_csv
    else:
        raise ValueError(f"Unknown split: {split_name}")

    # 1. Try to load cached data if requested
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Ensure attribute_ids are strings (parquet might infer types differently)
            if "attribute_ids" in df.columns:
                df["attribute_ids"] = df["attribute_ids"].astype(str)
            return df
        except Exception:
            # If load fails, proceed to process from scratch
            pass

    # 2. Process from scratch
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source metadata not found: {source_path}")

    df = pd.read_csv(source_path)

    # Handle NaNs in attribute_ids (empty labels)
    if "attribute_ids" in df.columns:
        df["attribute_ids"] = df["attribute_ids"].fillna("")
        # Ensure it's string type
        df["attribute_ids"] = df["attribute_ids"].astype(str)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to cache dataframe to {cache_path}: {e}")

    return df


class ArtworkDataset(Dataset):
    def __init__(self, df, transform=None, mode="train"):
        self.df = df
        self.transform = transform
        self.mode = mode
        self.input_dir = Config.input_dir
        self.num_classes = Config.num_classes
        self.img_size = Config.img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct file path
        # Metadata file_path is relative to input_dir (e.g. "train/xxx.png")
        image_path = os.path.join(self.input_dir, row["file_path"])

        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            # Robustness: Return black image if file read fails
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Default minimal transform
            t = A.Compose(
                [A.Resize(self.img_size, self.img_size), A.Normalize(), ToTensorV2()]
            )
            image = t(image=image)["image"]

        # Process Labels
        # Initialize target vector with zeros
        target = torch.zeros(self.num_classes, dtype=torch.float32)

        labels_str = row["attribute_ids"]
        # Check for valid label string
        if labels_str and labels_str.lower() != "nan" and labels_str.strip() != "":
            try:
                label_indices = [int(x) for x in labels_str.split()]
                for lbl_idx in label_indices:
                    if 0 <= lbl_idx < self.num_classes:
                        target[lbl_idx] = 1.0
            except ValueError:
                # In case of parsing error, target remains all zeros
                pass

        return image, target


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms based on the mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.img_size, Config.img_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test transforms (deterministic)
        return A.Compose(
            [
                A.Resize(Config.img_size, Config.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    # Set seed for reproducibility
    set_seed(Config.seed)

    # Load DataFrames
    train_df = load_dataset_metadata("train", load_cached_data=load_cached_data)
    val_df = load_dataset_metadata("val", load_cached_data=load_cached_data)
    test_df = load_dataset_metadata("test", load_cached_data=load_cached_data)

    # Debug mode subsampling
    if Config.debug:
        train_df = train_df.iloc[:1000]
        val_df = val_df.iloc[:500]
        test_df = test_df.iloc[:100]

    # Create Datasets
    train_dataset = ArtworkDataset(
        train_df, transform=get_transforms(mode="train"), mode="train"
    )
    val_dataset = ArtworkDataset(
        val_df, transform=get_transforms(mode="val"), mode="val"
    )
    test_dataset = ArtworkDataset(
        test_df, transform=get_transforms(mode="test"), mode="test"
    )

    # Create DataLoaders
    # Drop last for training to maintain consistent batch statistics
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
