import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.tokenizer import InChITokenizer


def load_dataframe(split_name, load_cached_data=True):
    """
    Loads metadata dataframe with caching logic.

    Args:
        split_name (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{split_name}_metadata.parquet")

    # Determine source path based on split
    if split_name == "train":
        csv_path = Config.TRAIN_METADATA_PATH
    elif split_name == "val":
        csv_path = Config.VAL_METADATA_PATH
    elif split_name == "test":
        csv_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split name: {split_name}")

    # Logic Flow
    df = None
    if load_cached_data:
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                # print(f"Loaded {split_name} metadata from cache: {cache_path}")
            except Exception:
                # print(f"Failed to load cache for {split_name}, reloading from source.")
                pass

    if df is None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Source metadata file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        # Save to cache
        df.to_parquet(cache_path, index=False)
        # print(f"Saved {split_name} metadata to cache: {cache_path}")

    return df


class ChemicalDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            tokenizer (InChITokenizer): Tokenizer instance.
            transform (albumentations.Compose): Image transformations.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        full_path = os.path.join(self.input_dir, file_path)

        # Load image
        # cv2.imread loads as BGR. We convert to RGB.
        # If the image is grayscale, cv2.IMREAD_COLOR converts it to BGR (3 channels) automatically.
        image = cv2.imread(full_path, cv2.IMREAD_COLOR)
        if image is None:
            # Handle missing image by creating a black image
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to basic tensor conversion
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.mode in ["train", "val"]:
            inchi_text = row["InChI"]
            # Tokenize
            seq = self.tokenizer.text_to_sequence(inchi_text)
            return image, torch.tensor(seq, dtype=torch.long)
        else:
            # Test mode
            image_id = row["image_id"]
            return image, image_id


def get_transforms(image_size):
    """
    Returns albumentations transforms for resizing and normalization.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(
                mean=Config.MEAN,
                std=Config.STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )


def get_dataloaders(
    load_cached_data=True, debug=False, batch_size=None, num_workers=None
):
    """
    Creates DataLoaders for train, val, and test splits.

    Args:
        load_cached_data (bool): Whether to use cached parquet files for metadata.
        debug (bool): If True, subsets the data for quick debugging.
        batch_size (int): Batch size override.
        num_workers (int): Num workers override.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Load DataFrames
    train_df = load_dataframe("train", load_cached_data)
    val_df = load_dataframe("val", load_cached_data)
    test_df = load_dataframe("test", load_cached_data)

    # Debug Subsetting
    if debug:
        train_df = train_df.iloc[:1000].reset_index(drop=True)
        val_df = val_df.iloc[:500].reset_index(drop=True)
        test_df = test_df.iloc[:100].reset_index(drop=True)
        # print("Debug mode enabled: Data subsetted.")

    # Tokenizer
    tokenizer = InChITokenizer()

    # Transforms
    transforms = get_transforms(Config.IMAGE_SIZE)

    # Datasets
    train_dataset = ChemicalDataset(
        train_df, tokenizer, transform=transforms, mode="train"
    )
    val_dataset = ChemicalDataset(val_df, tokenizer, transform=transforms, mode="val")
    test_dataset = ChemicalDataset(
        test_df, tokenizer, transform=transforms, mode="test"
    )

    # DataLoaders
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
