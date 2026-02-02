import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from library.config import Config
from library.utils import get_logger

# Initialize Logger
logger = get_logger("data_loader")


def load_and_cache_metadata(mode, load_cached_data=True):
    """
    Loads metadata from CSV or Parquet cache.

    Args:
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_metadata.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded {mode} metadata from cache: {cache_path}")
            return df
        except Exception as e:
            logger.warning(
                f"Failed to load cache {cache_path}: {e}. Reloading from source."
            )

    # 2. Load from source CSV
    if mode == "train":
        source_path = Config.TRAIN_METADATA
    elif mode == "val":
        source_path = Config.VAL_METADATA
    elif mode == "test":
        source_path = Config.TEST_METADATA
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Metadata file not found: {source_path}")

    df = pd.read_csv(source_path)
    logger.info(f"Loaded {mode} metadata from source: {source_path} ({len(df)} rows)")

    # 3. Apply Debug Sampling if enabled
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)
        logger.info(f"DEBUG mode: Sampled {mode} metadata to {len(df)} rows")

    # 4. Save to cache
    try:
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached {mode} metadata to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_path}: {e}")

    return df


class CatDogDataset(Dataset):
    """
    Dataset class for loading Dog vs Cat images.
    """

    def __init__(self, df, root_dir, transform=None, is_test=False):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        img_path = os.path.join(self.root_dir, row["filepath"])

        # Load image (RGB)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error to prevent crash,
            # though ideally data should be clean.
            image = Image.new("RGB", (256, 256), (0, 0, 0))

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get label
        if self.is_test:
            # For test data, we don't have labels. Return dummy or ID if needed.
            # Returning -1 as a placeholder.
            label = torch.tensor(-1, dtype=torch.float32)
        else:
            # Ensure label is float for BCEWithLogitsLoss
            label = torch.tensor(row["label"], dtype=torch.float32)

        return image, label


def get_transforms(img_size, mode="train"):
    """
    Generates the transformation pipeline based on the mode and image size.

    Args:
        img_size (int): Target image resolution (e.g., 224 or 256).
        mode (str): 'train' or 'val'/'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # Standard ImageNet Normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        # Augmentation Strategy: Context-Preservation + Photometric Noise
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (img_size, img_size),
                    scale=Config.AUG_SCALE,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ColorJitter(
                    brightness=Config.COLOR_JITTER_INTENSITY,
                    contrast=Config.COLOR_JITTER_INTENSITY,
                    saturation=Config.COLOR_JITTER_INTENSITY,
                    hue=Config.COLOR_JITTER_INTENSITY,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation/Test: Deterministic Resize
        # We resize directly to img_size to match the "Decoupled Pipelines" requirement
        # of explicit resolutions (256x256 or 224x224).
        return transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


def get_dataloaders(model_name, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        model_name (str): Key to look up model specs in Config.MODEL_SPECS.
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if model_name not in Config.MODEL_SPECS:
        raise ValueError(
            f"Unknown model_name: {model_name}. Available: {list(Config.MODEL_SPECS.keys())}"
        )

    specs = Config.MODEL_SPECS[model_name]
    img_size = specs["img_size"]
    batch_size = specs["batch_size"]

    logger.info(
        f"Initializing DataLoaders for {model_name} | Size: {img_size} | Batch: {batch_size}"
    )

    # 1. Load Metadata
    train_df = load_and_cache_metadata("train", load_cached_data)
    val_df = load_and_cache_metadata("val", load_cached_data)
    test_df = load_and_cache_metadata("test", load_cached_data)

    # 2. Create Transforms
    train_transform = get_transforms(img_size, mode="train")
    val_transform = get_transforms(img_size, mode="val")
    # Test transform is same as val (deterministic)
    test_transform = get_transforms(img_size, mode="val")

    # 3. Create Datasets
    train_dataset = CatDogDataset(
        train_df, Config.INPUT_DIR, transform=train_transform, is_test=False
    )
    val_dataset = CatDogDataset(
        val_df, Config.INPUT_DIR, transform=val_transform, is_test=False
    )
    test_dataset = CatDogDataset(
        test_df, Config.INPUT_DIR, transform=test_transform, is_test=True
    )

    # 4. Create DataLoaders
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Must be False to align with sorted IDs
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
