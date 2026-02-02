import torch
from torch.utils.data import Dataset
import numpy as np
import library.config as config
from library.data_utils import get_all_categorical_cols


class ManufacturingDataset(Dataset):
    def __init__(self, df, is_test=False):
        """
        PyTorch Dataset for the Manufacturing Control Data.

        Args:
            df (pd.DataFrame): The processed dataframe containing features and optionally targets.
            is_test (bool): Flag to indicate if this is the test set (no targets).
        """
        self.is_test = is_test

        # --- 1. Continuous Features ---
        # Select continuous columns defined in config
        cont_cols = config.ALL_CONTINUOUS_FEATURES
        # Convert to tensor (Float32 for neural networks)
        # We use .values to get the numpy array first, which is efficient
        self.continuous = torch.tensor(df[cont_cols].values, dtype=torch.float32)

        # --- 2. Categorical Features ---
        # Get the full list of categorical columns (original + decomposed)
        cat_cols = get_all_categorical_cols()
        # Convert to tensor (Long/Int64 is required for Embedding layers)
        self.categorical = torch.tensor(df[cat_cols].values, dtype=torch.long)

        # --- 3. Targets ---
        if not self.is_test and "target" in df.columns:
            # Convert target to Float32
            # Unsqueeze to shape (N, 1) to match BCEWithLogitsLoss expectation
            self.targets = torch.tensor(
                df["target"].values, dtype=torch.float32
            ).unsqueeze(1)
        else:
            self.targets = None

        # Optional: Store IDs if needed for tracking, though not used in training loop
        if "id" in df.columns:
            self.ids = df["id"].values
        else:
            self.ids = np.arange(len(df))

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.continuous)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Returns:
            dict: A dictionary containing:
                - 'continuous': Tensor of continuous features
                - 'categorical': Tensor of categorical features
                - 'target': Tensor of label (if available)
        """
        item = {
            "continuous": self.continuous[idx],
            "categorical": self.categorical[idx],
        }

        if self.targets is not None:
            item["target"] = self.targets[idx]

        return item
