import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from library.config import Config
from library.data_processor import DataProcessor


class TaxiDataset(Dataset):
    """
    PyTorch Dataset for the NYC Taxi Fare Prediction task.
    Loads processed data into memory as Tensors for efficient access.
    """

    def __init__(self, split="train", load_cached_data=True):
        """
        Args:
            split (str): One of 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load data from cache or re-process.
        """
        super().__init__()
        self.split = split

        # Initialize DataProcessor
        processor = DataProcessor()

        # Load all data splits
        # The processor handles caching logic internally based on the flag
        train_df, val_df, test_df = processor.process_data(
            load_cached_data=load_cached_data
        )

        # Select the appropriate dataframe
        if split == "train":
            self.df = train_df
        elif split == "val":
            self.df = val_df
        elif split == "test":
            self.df = test_df
        else:
            raise ValueError(
                f"Invalid split '{split}'. Must be 'train', 'val', or 'test'."
            )

        # Retrieve column definitions from processor to ensure consistency
        self.continuous_cols = processor.continuous_cols
        self.categorical_cols = processor.categorical_cols
        self.target_col = Config.TARGET_COL

        # Pre-convert data to PyTorch Tensors to speed up __getitem__
        # We use float32 for continuous features and int64 (Long) for categorical indices

        print(f"Converting {split} data to Tensors...")

        # 1. Continuous Features
        # Ensure values are float32
        cont_data = self.df[self.continuous_cols].values.astype(np.float32)
        self.continuous_features = torch.from_numpy(cont_data)

        # 2. Categorical Features (Spatial Indices + Time)
        # Ensure values are int64 for Embedding layers
        cat_data = self.df[self.categorical_cols].values.astype(np.int64)
        self.spatial_indices = torch.from_numpy(cat_data)

        # 3. Target (Fare Amount)
        # Test set does not have the target column
        if split != "test":
            target_data = self.df[self.target_col].values.astype(np.float32)
            self.targets = torch.from_numpy(target_data)
        else:
            self.targets = None

        # 4. Keys (for submission generation in test)
        if split == "test":
            self.keys = self.df["key"].values
        else:
            self.keys = None

        # Clean up dataframe to free memory, as we now hold tensors
        del self.df

        print(
            f"Dataset ({split}) loaded. Shape: {len(self.continuous_features)} samples."
        )

    def __len__(self):
        return len(self.continuous_features)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing:
            - continuous_features: Tensor of shape (num_continuous,)
            - spatial_indices: Tensor of shape (num_categorical,)
            - target: Scalar Tensor (if not test)
            - key: String (if test)
        """
        sample = {
            "continuous_features": self.continuous_features[idx],
            "spatial_indices": self.spatial_indices[idx],
        }

        if self.targets is not None:
            sample["target"] = self.targets[idx]

        if self.keys is not None:
            sample["key"] = self.keys[idx]

        return sample
