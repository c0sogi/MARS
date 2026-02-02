import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger(name="dataset")


def get_transforms(mode: str = "train"):
    """
    Returns the torchvision transforms for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: Composed transforms.
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Resize strategy: Resize shorter edge to 256, preserving aspect ratio.
    # Then CenterCrop to 224.
    common_transforms = [
        transforms.Resize(Config.RESIZE_SIZE),  # Integer argument scales short side
        transforms.CenterCrop(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ]

    if mode == "train":
        # Prepend augmentation for training
        # Only RandomHorizontalFlip as requested to preserve geometry
        train_transforms = [transforms.RandomHorizontalFlip(p=0.5)] + common_transforms
        return transforms.Compose(train_transforms)
    else:
        # Val and Test use deterministic transforms
        return transforms.Compose(common_transforms)


def process_data(load_cached_data: bool = True):
    """
    Loads metadata, processes label mappings, and handles caching using Parquet/Numpy.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (train_df, val_df, test_df, class_names)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")
    label_map_cache = os.path.join(cache_dir, "label_map.npy")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
            and os.path.exists(label_map_cache)
        ):
            logger.info("Loading processed data from cache...")
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                class_names = np.load(label_map_cache, allow_pickle=False)
                return train_df, val_df, test_df, class_names
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Reprocessing data.")
        else:
            logger.info("Cache files not found. Processing data from scratch...")
    else:
        logger.info("Forcing data reprocessing (load_cached_data=False)...")

    # Load raw metadata
    logger.info(f"Reading metadata from {Config.METADATA_DIR}")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Create Label Mapping
    # Combine unique breeds from train and val to ensure coverage (though stratified split should handle this)
    unique_breeds = sorted(
        list(set(train_df["breed"].unique()) | set(val_df["breed"].unique()))
    )
    class_names = np.array(unique_breeds)

    breed_to_idx = {breed: idx for idx, breed in enumerate(class_names)}

    # Map labels to integers
    train_df["label_idx"] = train_df["breed"].map(breed_to_idx)
    val_df["label_idx"] = val_df["breed"].map(breed_to_idx)

    # Verify mapping integrity
    if train_df["label_idx"].isnull().any() or val_df["label_idx"].isnull().any():
        raise ValueError(
            "Error in label mapping: Some breeds in data not found in unique breed list."
        )

    # Save to cache
    logger.info(f"Saving processed data to {cache_dir}")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)
    np.save(label_map_cache, class_names)

    return train_df, val_df, test_df, class_names


class DogDataset(Dataset):
    """
    Custom Dataset for Dog Breed Classification.
    """

    def __init__(self, df, transform=None, is_test=False, input_dir=Config.INPUT_DIR):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and optionally 'label_idx'.
            transform (callable, optional): Transform to be applied on a sample.
            is_test (bool): If True, returns only image and id (no label).
            input_dir (str): Root directory for images.
        """
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.input_dir = input_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative to input_dir (e.g., "train/id.jpg")
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image or handle error appropriately.
            # For this task, we assume data integrity based on analysis.
            image = Image.new("RGB", (Config.RESIZE_SIZE, Config.RESIZE_SIZE))

        # Apply Transforms
        if self.transform:
            image = self.transform(image)

        # Return data based on mode
        if self.is_test:
            # For test, return image and ID for submission creation
            return image, row["id"]
        else:
            # For train/val, return image and integer label
            label = row["label_idx"]
            return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug_subset_size=None,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached metadata.
        debug_subset_size (int, optional): If provided, reduces dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    # 1. Process Metadata
    train_df, val_df, test_df, class_names = process_data(
        load_cached_data=load_cached_data
    )

    # Debugging: Subset data if requested
    if debug_subset_size is not None:
        logger.info(f"DEBUG MODE: Subsetting data to {debug_subset_size} samples.")
        train_df = train_df.iloc[:debug_subset_size]
        val_df = val_df.iloc[:debug_subset_size]
        test_df = test_df.iloc[:debug_subset_size]

    # 2. Initialize Datasets
    train_dataset = DogDataset(
        train_df, transform=get_transforms(mode="train"), is_test=False
    )

    val_dataset = DogDataset(
        val_df, transform=get_transforms(mode="val"), is_test=False
    )

    test_dataset = DogDataset(
        test_df, transform=get_transforms(mode="test"), is_test=True
    )

    # 3. Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability in training
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

    logger.info(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    logger.info(f"Number of classes: {len(class_names)}")

    return train_loader, val_loader, test_loader, class_names
