import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from library.config import Config
from library.utils import seed_everything


def load_metadata(mode: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads metadata for the specified mode (train, val, test).
    Implements caching mechanism using Parquet files.

    Args:
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing filepaths and labels/ids.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_data.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Verify file existence for a sample to ensure cache isn't stale/broken
            if len(df) > 0 and os.path.exists(df.iloc[0]["filepath"]):
                return df
        except Exception:
            pass  # Fallback to loading from source

    # 2. Load from source metadata
    if mode == "train":
        source_path = Config.TRAIN_METADATA_PATH
    elif mode == "val":
        source_path = Config.VAL_METADATA_PATH
    elif mode == "test":
        source_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Metadata file not found: {source_path}")

    df = pd.read_csv(source_path)

    # Process filepaths to be absolute or relative to current working dir correctly
    # The metadata contains paths relative to ./input e.g., "train/cat.0.jpg"
    # We need to prepend Config.INPUT_DIR
    df["filepath"] = df["filepath"].apply(lambda x: os.path.join(Config.INPUT_DIR, x))

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_transforms(mode: str):
    """
    Returns the torchvision transforms for the specified mode.
    Enforces 256x256 resolution and Bicubic interpolation.

    Args:
        mode (str): 'train' or 'val'/'test'.
    """
    # ImageNet Mean and Std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose(
            [
                # Context-Preservation: RandomResizedCrop with scale (0.8, 1.0)
                # Interpolation: Bicubic (Critical for Swin/ConvNeXt)
                transforms.RandomResizedCrop(
                    size=(Config.IMG_SIZE, Config.IMG_SIZE),
                    scale=Config.CROP_SCALE,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # Photometric Noise
                transforms.ColorJitter(
                    brightness=Config.COLOR_JITTER_BRIGHTNESS,
                    contrast=Config.COLOR_JITTER_CONTRAST,
                    saturation=Config.COLOR_JITTER_SATURATION,
                    hue=Config.COLOR_JITTER_HUE,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation / Test
        return transforms.Compose(
            [
                # Resize strictly to 256x256
                transforms.Resize(
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class DogCatDataset(Dataset):
    """
    Custom Dataset for Dog vs Cat classification.
    """

    def __init__(self, df: pd.DataFrame, transform=None, mode: str = "train"):
        self.df = df
        self.transform = transform
        self.mode = mode

        # Pre-extract lists for faster access
        self.filepaths = self.df["filepath"].values

        if self.mode in ["train", "val"]:
            self.labels = self.df["label"].values.astype(float)
        else:
            self.ids = self.df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.filepaths[idx]

        # Load image
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            # Fallback for corrupted images (though analysis showed none)
            # Return a black image to prevent crash
            print(f"Warning: Could not load image {path}. Error: {e}")
            image = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.mode in ["train", "val"]:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            img_id = self.ids[idx]
            return image, img_id


def get_dataloaders(
    load_cached_data: bool = True,
    debug: bool = Config.DEBUG,
    debug_subset_size: int = Config.DEBUG_SUBSET_SIZE,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached metadata.
        debug (bool): If True, subsets the data for quick debugging.
        debug_subset_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Load Metadata
    train_df = load_metadata("train", load_cached_data)
    val_df = load_metadata("val", load_cached_data)
    test_df = load_metadata("test", load_cached_data)

    # Debug Mode: Subset data
    if debug:
        train_df = train_df.iloc[:debug_subset_size]
        val_df = val_df.iloc[:debug_subset_size]
        test_df = test_df.iloc[:debug_subset_size]

    # Initialize Datasets
    train_dataset = DogCatDataset(
        train_df, transform=get_transforms("train"), mode="train"
    )

    val_dataset = DogCatDataset(val_df, transform=get_transforms("val"), mode="val")

    test_dataset = DogCatDataset(test_df, transform=get_transforms("test"), mode="test")

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
