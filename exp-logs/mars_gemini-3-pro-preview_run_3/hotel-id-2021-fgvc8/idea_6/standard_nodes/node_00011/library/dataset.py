import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_transforms


class HotelDataset(Dataset):
    """
    Custom Dataset for Hotel ID recognition.
    """

    def __init__(self, df, img_root, transform=None, mode="train", label_to_idx=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            img_root (str): Root directory containing images.
            transform (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
            label_to_idx (dict): Mapping from hotel_id to class index (required for train/val).
        """
        self.df = df
        self.img_root = img_root
        self.transform = transform
        self.mode = mode
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        full_path = os.path.join(self.img_root, file_path)

        # Load image using OpenCV
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return logic based on mode
        if self.mode in ["train", "val"]:
            hotel_id = row["hotel_id"]
            # Map raw hotel_id to class index
            label = self.label_to_idx[hotel_id]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # Test mode: return image only
            return image


def get_label_encoder(load_cached_data=True):
    """
    Generates or loads the label encoder (unique classes).
    Implements strict caching logic using .npy format.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "label_encoder.npy")

    loaded = False
    classes = None

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_file):
        try:
            classes = np.load(cache_file)
            loaded = True
        except Exception:
            loaded = False

    # 2. If loading fails or not requested, compute from scratch
    if not loaded:
        # Always use full training set to define label space
        df_full = pd.read_csv(Config.TRAIN_CSV)
        classes = np.sort(df_full["hotel_id"].unique())
        np.save(cache_file, classes)

    # Create mapping dictionary
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    return classes, label_to_idx


def get_dataloaders(debug=Config.DEBUG, load_cached_data=True):
    """
    Factory function to create DataLoaders for train, val, and test.

    Args:
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): Whether to use cached label encoder.

    Returns:
        train_loader, val_loader, test_loader, classes
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Apply Debug Sampling
    if debug:
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Get Label Encoder
    classes, label_to_idx = get_label_encoder(load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = HotelDataset(
        df=train_df,
        img_root=Config.IMG_ROOT_DIR,
        transform=get_transforms(img_size=Config.IMG_SIZE, mode="train"),
        mode="train",
        label_to_idx=label_to_idx,
    )

    val_dataset = HotelDataset(
        df=val_df,
        img_root=Config.IMG_ROOT_DIR,
        transform=get_transforms(img_size=Config.IMG_SIZE, mode="valid"),
        mode="val",
        label_to_idx=label_to_idx,
    )

    test_dataset = HotelDataset(
        df=test_df,
        img_root=Config.IMG_ROOT_DIR,
        transform=get_transforms(img_size=Config.IMG_SIZE, mode="test"),
        mode="test",
    )

    # Create DataLoaders
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

    return train_loader, val_loader, test_loader, classes
