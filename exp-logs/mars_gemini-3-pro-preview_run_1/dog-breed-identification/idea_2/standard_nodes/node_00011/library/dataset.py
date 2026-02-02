import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility across the module
set_seed(Config.SEED)


class DogDataset(Dataset):
    """
    Custom Dataset for Dog Breed Classification.
    Loads images based on metadata paths and applies geometric-preserving transforms.
    """

    def __init__(self, df, root_dir, transform=None, mode="train", class_to_idx=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path', 'breed' (for train/val), and 'id'.
            root_dir (str): Root directory containing the images (Config.INPUT_DIR).
            transform (callable, optional): Transform pipeline.
            mode (str): 'train', 'val', or 'test'.
            class_to_idx (dict, optional): Mapping from breed name to integer index. Required for train/val.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative (e.g., "train/id.jpg")
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for robustness (though metadata verification passed)
            print(f"Warning: Could not load image {img_path}. Error: {e}")
            image = Image.new("RGB", (Config.RESIZE_SIZE, Config.RESIZE_SIZE))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.mode in ["train", "val"]:
            # Return image and integer label
            breed = row["breed"]
            label = self.class_to_idx[breed]
            return image, label
        else:
            # For test, return image and ID string for submission mapping
            img_id = row["id"]
            return image, img_id


def get_transforms(split):
    """
    Returns the transformation pipeline for a given split.
    Strictly implements Resize(256) -> CenterCrop(224) for geometric integrity.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Base transforms for all splits
    common_transforms = [
        transforms.Resize(Config.RESIZE_SIZE),  # Resizes smaller edge to 256
        transforms.CenterCrop(Config.CROP_SIZE),  # Crops center 224x224
    ]

    if split == "train":
        # Training: Add RandomHorizontalFlip
        t_list = common_transforms + [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    else:
        # Validation/Test: Deterministic
        t_list = common_transforms + [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

    return transforms.Compose(t_list)


def _get_metadata(split, load_cached_data):
    """
    Internal helper to load metadata from CSV or Cache.
    Handles caching logic and Debug subsetting.
    """
    # Determine file paths
    if split == "train":
        csv_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.CACHE_TRAIN_PATH
    elif split == "val":
        csv_path = Config.VAL_METADATA_PATH
        cache_path = Config.CACHE_VAL_PATH
    elif split == "test":
        csv_path = Config.TEST_METADATA_PATH
        cache_path = Config.CACHE_TEST_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # If in DEBUG mode, subset the loaded full data
            if Config.DEBUG:
                df = df.head(Config.DEBUG_SUBSET_SIZE)
            return df
        except Exception:
            pass  # Fallback to computing from scratch

    # 2. Load from source CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # 3. Save full dataframe to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    # 4. Apply Debug Subset if needed
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    return df


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test splits.

    Args:
        load_cached_data (bool): Whether to attempt loading cached metadata.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_list)
    """
    # Load DataFrames
    train_df = _get_metadata("train", load_cached_data)
    val_df = _get_metadata("val", load_cached_data)
    test_df = _get_metadata("test", load_cached_data)

    # Determine Class Mapping
    # We must ensure the class mapping is consistent and complete.
    # If in DEBUG mode, the subsetted train_df might miss some classes.
    # We read the full training metadata to establish the canonical class list.
    if Config.DEBUG:
        full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        classes = sorted(full_train_df["breed"].unique().tolist())
    else:
        classes = sorted(train_df["breed"].unique().tolist())

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # Create Datasets
    train_dataset = DogDataset(
        df=train_df,
        root_dir=Config.INPUT_DIR,
        transform=get_transforms("train"),
        mode="train",
        class_to_idx=class_to_idx,
    )

    val_dataset = DogDataset(
        df=val_df,
        root_dir=Config.INPUT_DIR,
        transform=get_transforms("val"),
        mode="val",
        class_to_idx=class_to_idx,
    )

    test_dataset = DogDataset(
        df=test_df,
        root_dir=Config.INPUT_DIR,
        transform=get_transforms("test"),
        mode="test",
        class_to_idx=None,  # Not needed for test
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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

    return train_loader, val_loader, test_loader, classes
