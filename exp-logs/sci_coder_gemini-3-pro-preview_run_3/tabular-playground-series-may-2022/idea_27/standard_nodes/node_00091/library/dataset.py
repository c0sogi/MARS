import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ManufacturingDataset(Dataset):
    """
    Dataset class for the Manufacturing Control Data.
    Wraps processed continuous and categorical features and targets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cont_cols: list,
        cat_cols: list,
        target_col: str = None,
        max_samples: int = None,
    ):
        """
        Args:
            df (pd.DataFrame): The dataframe containing the processed data.
            cont_cols (list): List of column names for continuous features.
            cat_cols (list): List of column names for categorical features.
            target_col (str, optional): Name of the target column. Defaults to None.
            max_samples (int, optional): If set, truncates the dataset to the first `max_samples`.
                                         Useful for debugging and quick iterations.
        """
        # Debugging: Truncate dataset if max_samples is provided
        if max_samples is not None and max_samples > 0:
            df = df.iloc[:max_samples].copy()

        # Store continuous features as float32
        # We extract values once to avoid pandas overhead during iteration
        self.cont_features = df[cont_cols].values.astype(np.float32)

        # Store categorical features as int64 (required for PyTorch Embedding layers)
        self.cat_features = df[cat_cols].values.astype(np.int64)

        # Store target if available
        self.target = None
        if target_col is not None and target_col in df.columns:
            self.target = df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.cont_features)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the tensors for the given index.

        Returns:
            dict: {
                'continuous': torch.FloatTensor,
                'categorical': torch.LongTensor,
                'target': torch.FloatTensor (if available)
            }
        """
        # Convert numpy rows to tensors
        # torch.from_numpy shares memory and is efficient for numpy arrays
        x_cont = torch.from_numpy(self.cont_features[idx])
        x_cat = torch.from_numpy(self.cat_features[idx])

        sample = {"continuous": x_cont, "categorical": x_cat}

        if self.target is not None:
            # Target is a scalar, so we construct a 0-d tensor
            sample["target"] = torch.tensor(self.target[idx], dtype=torch.float32)

        return sample
