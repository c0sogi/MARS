import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from library.augmentations import get_train_transforms, get_valid_transforms

# Constants
CACHE_DIR = "./working/idea_3/"
METADATA_DIR = "./metadata/"
INPUT_DIR = "./input"
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def get_dataframe(mode, load_cached_data=True):
    """
    Loads the dataframe for the specific mode (train/val/test).
    Implements the required caching mechanism using Parquet.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{mode}_cache.parquet")
    metadata_path = os.path.join(METADATA_DIR, f"{mode}_metadata.csv")

    # 1. Try to load from cache if requested
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, fall through to process from scratch
            pass

    # 2. Process from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Ensure full paths are correct (metadata has relative paths)
    # We don't need to rewrite the column, but we verify the logic here.
    # The Dataset class will construct the full path.

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

    return df


class AppleDataset(Dataset):
    def __init__(
        self,
        mode,
        transform=None,
        data_dir=INPUT_DIR,
        load_cached_data=True,
        debug_subset_size=None,
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Transformations to apply.
            data_dir (str): Root directory for images.
            load_cached_data (bool): Whether to use cached metadata.
            debug_subset_size (int, optional): If set, limits dataset size for debugging.
        """
        self.mode = mode
        self.data_dir = data_dir
        self.transform = transform

        # Load Dataframe
        self.df = get_dataframe(mode, load_cached_data=load_cached_data)

        # Debugging: Subset
        if debug_subset_size is not None:
            self.df = self.df.iloc[:debug_subset_size].reset_index(drop=True)

        # Pre-extract paths and labels for faster access
        self.image_ids = self.df["image_id"].values
        self.file_paths = self.df["file_path"].values

        if self.mode != "test":
            self.labels = self.df[TARGET_COLS].values.astype(np.float32)
        else:
            # Dummy labels for test set
            self.labels = np.zeros((len(self.df), len(TARGET_COLS)), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full path
        rel_path = self.file_paths[idx]
        img_path = os.path.join(self.data_dir, rel_path)

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or could not be read: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to ToTensor if no transform provided (though usually provided)
            # This part assumes the user provides a transform that outputs a tensor.
            # If not, we manually convert.
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Get Label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label

    def get_labels(self):
        """Helper to get all labels (useful for weighted sampling or metrics)"""
        return self.labels

    def get_image_ids(self):
        """Helper to get all image IDs"""
        return self.image_ids
