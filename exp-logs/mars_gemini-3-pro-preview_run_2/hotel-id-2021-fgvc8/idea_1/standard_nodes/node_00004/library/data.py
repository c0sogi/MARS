import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything

# =========================================================================
# Dataset Class
# =========================================================================


class HotelDataset(Dataset):
    """
    PyTorch Dataset for Hotel ID classification.
    Reads images from disk, applies transforms, and returns (image, label/id).
    """

    def __init__(
        self,
        df,
        transform=None,
        class_to_idx=None,
        is_test=False,
        root_dir=Config.INPUT_DIR,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path' and 'hotel_id'.
            transform (albumentations.Compose): Transformations to apply to the image.
            class_to_idx (dict): Mapping from original hotel_id to integer index (0..N-1).
            is_test (bool): If True, returns (image, image_filename). If False, returns (image, label_idx).
            root_dir (str): Root directory containing the images.
        """
        self.df = df
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.is_test = is_test
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative, e.g., "train_images/0/xyz.jpg"
        image_path = os.path.join(self.root_dir, row["file_path"])

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            # Handle missing or corrupt images by creating a black image
            # This ensures the dataloader doesn't crash
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.is_test:
            # For test set, return image and the image filename (ID)
            return image, row["image"]
        else:
            # For train/val, return image and the mapped integer label
            original_id = row["hotel_id"]
            label = self.class_to_idx[original_id]
            return image, torch.tensor(label, dtype=torch.long)


# =========================================================================
# Transforms
# =========================================================================


def get_transforms(data_split):
    """
    Returns the Albumentations transformations for a specific data split.

    Args:
        data_split (str): 'train', 'val', or 'test'.
    """
    # Standard ImageNet normalization stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Resize slightly larger than target, then crop
    resize_dim = 256
    crop_dim = Config.IMAGE_SIZE  # 224

    if data_split == "train":
        return A.Compose(
            [
                A.Resize(height=resize_dim, width=resize_dim),
                A.RandomCrop(height=crop_dim, width=crop_dim),
                A.HorizontalFlip(p=0.5),
                # Add more augmentations here if needed (e.g., ColorJitter, CoarseDropout)
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Deterministic Center Crop
        return A.Compose(
            [
                A.Resize(height=resize_dim, width=resize_dim),
                A.CenterCrop(height=crop_dim, width=crop_dim),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


# =========================================================================
# Data Loading & Processing
# =========================================================================


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    Handles label encoding caching.

    Args:
        load_cached_data (bool): Whether to load cached class mappings.
        debug (bool): If True, subsamples the dataset for quick debugging.

    Returns:
        train_loader, val_loader, test_loader, classes (list of original IDs)
    """
    seed_everything(Config.SEED)

    # Ensure cache directory exists
    os.makedirs(Config.IDEA_DIR, exist_ok=True)
    cache_path = os.path.join(Config.IDEA_DIR, "classes.parquet")

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Handle Label Encoding (Caching)
    classes = None

    if load_cached_data and os.path.exists(cache_path):
        try:
            # Load classes from parquet
            classes_df = pd.read_parquet(cache_path)
            classes = classes_df["hotel_id"].values
            print(f"Loaded {len(classes)} classes from cache.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")
            classes = None

    if classes is None:
        # Compute unique classes from training data
        # We sort them to ensure deterministic mapping 0..N-1
        classes = np.sort(train_df["hotel_id"].unique())

        # Save to cache
        pd.DataFrame({"hotel_id": classes}).to_parquet(cache_path, index=False)
        print(f"Computed and cached {len(classes)} classes.")

    # Create mapping: original_id -> index
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    # Validate that validation set classes are covered
    # (The metadata generation script ensures this, but good to be safe)
    # Note: If val has classes not in train, this would crash.
    # The metadata script puts singletons in train, so train is a superset.

    # 3. Debug Mode (Subsampling)
    if debug:
        print("Debug mode: Subsampling datasets...")
        train_df = train_df.sample(
            n=min(2000, len(train_df)), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(500, len(val_df)), random_state=Config.SEED
        ).reset_index(drop=True)
        # Keep test full or sample? Usually debug implies checking pipeline speed.
        test_df = test_df.sample(
            n=min(100, len(test_df)), random_state=Config.SEED
        ).reset_index(drop=True)

    # 4. Create Datasets
    train_dataset = HotelDataset(
        train_df,
        transform=get_transforms("train"),
        class_to_idx=class_to_idx,
        is_test=False,
    )

    val_dataset = HotelDataset(
        val_df,
        transform=get_transforms("val"),
        class_to_idx=class_to_idx,
        is_test=False,
    )

    test_dataset = HotelDataset(
        test_df,
        transform=get_transforms("test"),
        class_to_idx=None,  # Not needed for test
        is_test=True,
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to maintain batch stats stability
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

    return train_loader, val_loader, test_loader, classes
