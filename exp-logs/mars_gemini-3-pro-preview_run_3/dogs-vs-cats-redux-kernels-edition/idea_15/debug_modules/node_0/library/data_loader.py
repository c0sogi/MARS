import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from library import config


def _load_metadata(filename, load_cached_data=True):
    """
    Loads metadata dataframe with caching logic.

    Args:
        filename (str): The name of the csv file in metadata dir (e.g., 'train.csv').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    cache_path = os.path.join(
        config.WORKING_DIR, f"{filename.replace('.csv', '')}_meta.parquet"
    )
    csv_path = os.path.join(config.METADATA_DIR, filename)

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to processing from scratch

    # 2. IF loading fails OR load_cached_data is False:
    # Compute/process the data from scratch.
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Save the result to the cache directory
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path)

    # 3. Return the data.
    return df


def get_transforms(img_size, mode="train"):
    """
    Constructs the transformation pipeline based on the mode and image size.

    Args:
        img_size (int): The target spatial dimension (H=W).
        mode (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if mode == "train":
        return transforms.Compose(
            [
                # Context-Preserving Augmentation: Scale 0.8-1.0 ensures subject is kept
                transforms.RandomResizedCrop(
                    (img_size, img_size),
                    scale=(0.8, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                # Intensity >= 0.2 for lighting variance
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        # Val/Test: Resize to target directly using Bicubic
        return transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )


class CatDogDataset(Dataset):
    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transform (callable, optional): Transform to apply to images.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path
        rel_path = row["filepath"]
        full_path = os.path.join(config.INPUT_DIR, rel_path)

        # Load image
        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt images (though analysis showed 0 missing)
            # Return a black image or handle gracefully.
            # Given constraints, we assume valid data as per analysis.
            image = Image.new("RGB", (256, 256))

        if self.transform:
            image = self.transform(image)

        if self.mode == "test":
            # Return image and ID for submission
            return image, row["id"]
        else:
            # Return image and label (float for BCEWithLogitsLoss usually,
            # but standard is often Long for CrossEntropy.
            # Config mentions BCEWithLogitsLoss, so Float is safer for target).
            label = torch.tensor(row["label"], dtype=torch.float32)
            return image, label


def get_dataloaders(model_key, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        model_key (str): Key to look up model specs in config (e.g., 'resnet').
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if model_key not in config.MODEL_SPECS:
        raise ValueError(
            f"Unknown model key: {model_key}. Available: {list(config.MODEL_SPECS.keys())}"
        )

    specs = config.MODEL_SPECS[model_key]
    img_size = specs["img_size"]

    # Load Metadata
    train_df = _load_metadata("train.csv", load_cached_data=load_cached_data)
    val_df = _load_metadata("val.csv", load_cached_data=load_cached_data)
    test_df = _load_metadata("test.csv", load_cached_data=load_cached_data)

    # Debug Subsetting
    if config.DEBUG:
        subset_size = config.SUBSET_SIZE if config.SUBSET_SIZE else 100
        train_df = train_df.head(subset_size)
        val_df = val_df.head(subset_size)
        test_df = test_df.head(subset_size)

    # Transforms
    train_transform = get_transforms(img_size, mode="train")
    val_transform = get_transforms(img_size, mode="val")
    # Test transform is same as val usually (deterministic resize)
    test_transform = get_transforms(img_size, mode="test")

    # Datasets
    train_dataset = CatDogDataset(train_df, transform=train_transform, mode="train")
    val_dataset = CatDogDataset(val_df, transform=val_transform, mode="val")
    test_dataset = CatDogDataset(test_df, transform=test_transform, mode="test")

    # DataLoaders
    # Use persistent_workers=True if num_workers > 0 to speed up training
    num_workers = config.NUM_WORKERS
    persistent = True if num_workers > 0 else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader
