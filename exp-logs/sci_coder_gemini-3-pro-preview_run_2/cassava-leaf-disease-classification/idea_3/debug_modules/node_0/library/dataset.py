import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from library.config import CFG


def load_metadata(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads the metadata dataframe for a given split, implementing the required caching logic.

    Args:
        split (str): The data split ('train', 'val', or 'test').
        load_cached_data (bool): Whether to attempt loading from the cache.

    Returns:
        pd.DataFrame: The processed metadata dataframe.
    """
    # Define cache file path
    cache_filename = f"cached_{split}_metadata.parquet"
    cache_path = os.path.join(CFG.output_dir, cache_filename)

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If loading fails, proceed to compute from scratch
            pass

    # 2. IF loading fails OR load_cached_data is False: Compute/process data
    if split == "train":
        csv_path = CFG.train_csv
    elif split == "val":
        csv_path = CFG.val_csv
    elif split == "test":
        csv_path = CFG.test_csv
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', or 'test'.")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found at {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Apply subsetting if defined in configuration (for debugging)
    if split == "train" and CFG.train_subset_size is not None:
        df = df.iloc[: CFG.train_subset_size]
    elif split == "val" and CFG.val_subset_size is not None:
        df = df.iloc[: CFG.val_subset_size]

    # Save the result to the cache directory
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class CassavaDataset(Dataset):
    """
    PyTorch Dataset for Cassava Leaf Disease Classification.
    Handles loading images via PIL and applying transformations.
    """

    def __init__(self, split: str, transform=None, load_cached_data: bool = True):
        """
        Args:
            split (str): The data split ('train', 'val', or 'test').
            transform (callable, optional): The transformation pipeline to apply to images.
            load_cached_data (bool): Whether to use cached metadata.
        """
        self.split = split
        self.transform = transform

        # Load metadata using the caching helper
        self.df = load_metadata(split, load_cached_data=load_cached_data)

        # Extract columns to arrays for faster access during iteration
        self.file_paths = self.df["file_path"].values
        self.labels = self.df["label"].values

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image_tensor, label_tensor)
        """
        # Get relative path and label
        rel_path = self.file_paths[idx]
        label_val = self.labels[idx]

        # Construct full path
        img_path = os.path.join(CFG.input_root, rel_path)

        # Load image using PIL
        # .convert("RGB") ensures consistency (3 channels) even if input is grayscale or RGBA
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image at {img_path}: {e}")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        target = torch.tensor(label_val, dtype=torch.long)

        return image, target
