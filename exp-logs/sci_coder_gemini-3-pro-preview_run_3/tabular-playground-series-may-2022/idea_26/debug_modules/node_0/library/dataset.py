import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Handles storage and retrieval of normalized continuous features,
    integer-encoded categorical features, and binary targets.
    """

    def __init__(self, df, cat_cols, cont_cols, target_col="target", is_test=False):
        """
        Args:
            df (pd.DataFrame): The processed dataframe containing features and targets.
            cat_cols (list of str): List of column names for categorical features.
            cont_cols (list of str): List of column names for continuous features.
            target_col (str): Name of the target column.
            is_test (bool): Flag indicating if this is the test set (no targets).
        """
        self.is_test = is_test

        # Convert continuous features to float32 numpy array for speed
        # We use .values ensures we are working with numpy arrays, avoiding pandas overhead in __getitem__
        self.continuous_data = df[cont_cols].values.astype(np.float32)

        # Convert categorical features to int64 (long) numpy array
        self.categorical_data = df[cat_cols].values.astype(np.int64)

        # Handle targets if not in test mode
        if not self.is_test:
            if target_col in df.columns:
                self.targets = df[target_col].values.astype(np.float32)
            else:
                raise ValueError(
                    f"Target column '{target_col}' not found in dataframe."
                )
        else:
            self.targets = None

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.continuous_data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'continuous': Tensor of shape (num_cont_features,)
                - 'categorical': Tensor of shape (num_cat_features,)
                - 'target': Tensor of shape (1,) (only if not is_test)
        """
        # Fetch data from numpy arrays and convert to tensors
        # Creating tensors from numpy arrays is efficient
        cont_tensor = torch.tensor(self.continuous_data[idx], dtype=torch.float32)
        cat_tensor = torch.tensor(self.categorical_data[idx], dtype=torch.long)

        sample = {"continuous": cont_tensor, "categorical": cat_tensor}

        if not self.is_test:
            # Target needs to be shape (1,) for BCEWithLogitsLoss usually,
            # or just a scalar depending on implementation.
            # Keeping it consistent as a 1D tensor is safest for unsqueeze logic later if needed.
            target_val = self.targets[idx]
            sample["target"] = torch.tensor([target_val], dtype=torch.float32)

        return sample
