import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Define the list of binary metadata features based on the dataset description
META_FEATURES = [
    "Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]


class PetDataset(Dataset):
    """
    Dataset class for loading Pet Pawpularity images and metadata.
    """

    def __init__(self, df, image_dir=Config.INPUT_DIR, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata and file paths.
            image_dir (str): Root directory for images.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # row['file_path'] is relative, e.g., "train/id.jpg"
        img_path = os.path.join(self.image_dir, row["file_path"])

        # Load image using OpenCV
        img = cv2.imread(img_path)

        # Handle missing images (robustness)
        if img is None:
            # Create a blank image if file is missing/corrupt to prevent crash
            img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size
        img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        # Normalize (Manual implementation of ImageNet normalization)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array(Config.IMAGENET_MEAN)) / np.array(Config.IMAGENET_STD)

        # Convert HWC to CHW format for PyTorch
        img = img.transpose(2, 0, 1)
        img_tensor = torch.tensor(img, dtype=torch.float32)

        # Extract binary meta features
        # Ensure columns exist, fill with 0 if missing (robustness)
        meta_vals = []
        for col in META_FEATURES:
            if col in row:
                meta_vals.append(row[col])
            else:
                meta_vals.append(0)
        meta_tensor = torch.tensor(meta_vals, dtype=torch.float32)

        # Extract Target
        target = 0.0
        if "Pawpularity" in row:
            target = row["Pawpularity"]
        target_tensor = torch.tensor(target, dtype=torch.float32)

        # Extract ID
        sample_id = str(row["Id"])

        return img_tensor, meta_tensor, target_tensor, sample_id


def get_dataloaders(train_df=None, val_df=None, test_df=None):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        train_df (pd.DataFrame, optional): Training dataframe.
        val_df (pd.DataFrame, optional): Validation dataframe.
        test_df (pd.DataFrame, optional): Test dataframe.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders.
    """
    # Load dataframes from metadata files if not provided
    if train_df is None and os.path.exists(Config.TRAIN_META_PATH):
        train_df = pd.read_csv(Config.TRAIN_META_PATH)

    if val_df is None and os.path.exists(Config.VAL_META_PATH):
        val_df = pd.read_csv(Config.VAL_META_PATH)

    if test_df is None and os.path.exists(Config.TEST_META_PATH):
        test_df = pd.read_csv(Config.TEST_META_PATH)

    # Apply Debug Sampling
    if Config.DEBUG:
        print(f"DEBUG mode enabled. Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        if train_df is not None:
            train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        if val_df is not None:
            val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        if test_df is not None:
            test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    loaders = {}

    # Common DataLoader kwargs
    dl_kwargs = {
        "batch_size": Config.BATCH_SIZE,
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": True if Config.DEVICE == "cuda" else False,
    }

    if train_df is not None:
        train_ds = PetDataset(train_df, mode="train")
        loaders["train"] = DataLoader(train_ds, shuffle=True, **dl_kwargs)

    if val_df is not None:
        val_ds = PetDataset(val_df, mode="val")
        loaders["val"] = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    if test_df is not None:
        test_ds = PetDataset(test_df, mode="test")
        loaders["test"] = DataLoader(test_ds, shuffle=False, **dl_kwargs)

    return loaders
