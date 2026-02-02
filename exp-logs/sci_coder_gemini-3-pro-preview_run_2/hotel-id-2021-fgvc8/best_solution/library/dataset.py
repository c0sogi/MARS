import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import seed_everything


class HotelDataset(Dataset):
    """
    PyTorch Dataset for Hotel Identification.
    Reads images from disk, applies augmentations, and returns tensors.
    """

    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_path, label_idx).
            transform (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transform = transform
        self.mode = mode
        self.data_root = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # file_path in metadata is relative to input dir (e.g., "train_images/1/abc.jpg")
        file_path = os.path.join(self.data_root, row["file_path"])

        # Read image using OpenCV
        image = cv2.imread(file_path)

        # Handle potential read errors (though metadata validation passed)
        if image is None:
            # Fallback: create a black image to prevent crash, print warning
            print(f"Warning: Could not read image at {file_path}")
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided (Resize + ToTensor)
            # This fallback ensures we always return a tensor
            T = A.Compose(
                [
                    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    A.Normalize(mean=Config.MEAN, std=Config.STD),
                    ToTensorV2(),
                ]
            )
            image = T(image=image)["image"]

        # Return data based on mode
        if self.mode in ["train", "val"]:
            # Return image and integer label
            label = torch.tensor(row["label_idx"], dtype=torch.long)
            return image, label
        else:
            # Test mode: Return image and original image filename (for submission mapping)
            return image, row["image"]


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms based on the mode.

    Args:
        mode (str): 'train' or 'val'/'test'.

    Returns:
        A.Compose: Composed transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                # ColorJitter as per Idea 3 description
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                # CoarseDropout as per Idea 3 description
                A.CoarseDropout(
                    max_holes=8,
                    max_height=Config.IMAGE_SIZE // 10,
                    max_width=Config.IMAGE_SIZE // 10,
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


def get_label_mapping(load_cached_data=True):
    """
    Generates or loads the mapping between hotel_id and integer class index.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        tuple: (id_to_idx, idx_to_id) dictionaries.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "classes.npy")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading label mapping from {cache_path}")
        classes = np.load(cache_path, allow_pickle=False)
    else:
        # 2. Compute from scratch
        print("Computing label mapping from training metadata...")
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

        # Get unique hotel IDs and sort them to ensure deterministic mapping
        classes = np.unique(train_df["hotel_id"].values)
        classes.sort()

        # Save to cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(cache_path, classes)
        print(f"Saved label mapping to {cache_path}")

    # Create dictionaries
    # classes array: index -> hotel_id
    idx_to_id = {i: hotel_id for i, hotel_id in enumerate(classes)}
    id_to_idx = {hotel_id: i for i, hotel_id in enumerate(classes)}

    return id_to_idx, idx_to_id


def get_dataloaders(debug=False):
    """
    Prepares DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, subsets data for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    seed_everything(Config.SEED)

    # --- Load Metadata ---
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # --- Label Mapping ---
    # Ensure we use the cached mapping or generate it consistently
    id_to_idx, idx_to_id = get_label_mapping(load_cached_data=True)
    num_classes = len(id_to_idx)

    # Map hotel_id to label_idx for Train and Val
    # Note: test_df 'hotel_id' column is a placeholder string, so we don't map it.

    # Filter out any potential new classes in Val that weren't in Train
    # (Metadata validation ensures this doesn't happen, but good for safety)
    train_df["label_idx"] = train_df["hotel_id"].map(id_to_idx)
    val_df["label_idx"] = val_df["hotel_id"].map(id_to_idx)

    # Drop any rows where mapping failed (should be 0 based on checks)
    train_df = train_df.dropna(subset=["label_idx"])
    val_df = val_df.dropna(subset=["label_idx"])

    train_df["label_idx"] = train_df["label_idx"].astype(int)
    val_df["label_idx"] = val_df["label_idx"].astype(int)

    # --- Debug Mode ---
    if debug or Config.DEBUG:
        print(f"Debug mode enabled. Subsampling {Config.DEBUG_SAMPLE_SIZE} samples.")
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # --- Datasets ---
    train_dataset = HotelDataset(
        train_df, transform=get_transforms(mode="train"), mode="train"
    )

    val_dataset = HotelDataset(val_df, transform=get_transforms(mode="val"), mode="val")

    test_dataset = HotelDataset(
        test_df, transform=get_transforms(mode="test"), mode="test"
    )

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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

    print(f"DataLoaders created:")
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    print(f"  Num Classes: {num_classes}")

    return train_loader, val_loader, test_loader, num_classes
