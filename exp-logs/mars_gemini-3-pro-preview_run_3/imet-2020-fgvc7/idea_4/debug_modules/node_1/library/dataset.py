import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def load_processed_dataframe(mode, load_cached_data=True):
    """
    Loads the metadata dataframe for a specific mode (train/val/test).
    Implements caching using Parquet files to store processed dataframes
    (where attribute_ids strings are converted to lists).
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"cached_{mode}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, proceed to process from scratch
            pass

    # 2. Process from scratch
    if mode == "train":
        csv_path = Config.TRAIN_CSV
    elif mode == "val":
        csv_path = Config.VAL_CSV
    elif mode == "test":
        csv_path = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse attribute_ids: "0 1 2" -> [0, 1, 2]
    # Handle NaNs by converting to empty string first
    df["attribute_ids"] = df["attribute_ids"].fillna("")

    # Function to safe convert string to list of ints
    def parse_ids(x):
        if not x.strip():
            return np.array([], dtype=int)
        return np.array([int(i) for i in x.split()], dtype=int)

    # We store as numpy arrays inside the dataframe cells for parquet compatibility/efficiency
    df["parsed_attributes"] = df["attribute_ids"].apply(parse_ids)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df


class ArtworkDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.input_dir = Config.INPUT_DIR
        self.num_classes = Config.NUM_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["id"]
        file_path = row["file_path"]

        # Load Image
        full_path = os.path.join(self.input_dir, file_path)
        image = cv2.imread(full_path)

        if image is None:
            # Fallback for missing/corrupt images: return black image
            # This prevents crashing during training
            image = np.zeros(
                (Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Create Target (Multi-hot encoding)
        target = torch.zeros(self.num_classes, dtype=torch.float32)

        # In test mode, we might not have valid labels, but we still return a dummy target
        # parsed_attributes is a numpy array of integers
        attr_ids = row["parsed_attributes"]

        if len(attr_ids) > 0:
            # Ensure indices are within bounds
            valid_ids = attr_ids[attr_ids < self.num_classes]
            target[valid_ids] = 1.0

        return image, target, image_id


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.
    """
    img_size = Config.IMG_SIZE

    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    debug=False,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, subsets the data for quick debugging.
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of worker processes.
        load_cached_data (bool): Whether to use cached dataframes.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load DataFrames
    train_df = load_processed_dataframe("train", load_cached_data)
    val_df = load_processed_dataframe("val", load_cached_data)
    test_df = load_processed_dataframe("test", load_cached_data)

    # Debug Subsampling
    if debug:
        debug_size = Config.DEBUG_SAMPLE_SIZE
        train_df = train_df.iloc[:debug_size]
        val_df = val_df.iloc[:debug_size]
        test_df = test_df.iloc[:debug_size]

    # Create Datasets
    train_dataset = ArtworkDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = ArtworkDataset(val_df, transforms=get_transforms("val"), mode="val")
    test_dataset = ArtworkDataset(
        test_df, transforms=get_transforms("test"), mode="test"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

    return train_loader, val_loader, test_loader
