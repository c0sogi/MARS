import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from library.config import Config


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control Data.

    Prepares data for the Hybrid Transformer-Funnel architecture by separating
    categorical sequences (for the Transformer branch) and continuous features
    (for the Funnel MLP branch).
    """

    def __init__(self, df: pd.DataFrame, is_test: bool = False):
        """
        Args:
            df (pd.DataFrame): The preprocessed dataframe containing features and optionally targets.
            is_test (bool): Flag indicating if this is the test set (no targets).
        """
        self.is_test = is_test

        # Define feature groups based on Config and Preprocessing logic
        # Categorical: 10 decomposed chars + f_29 + f_30
        self.cat_cols = [f"ch_{i}" for i in range(10)] + ["f_29", "f_30"]

        # Continuous: Normalized features defined in Config
        self.cont_cols = Config.CONTINUOUS_FEATURE_NAMES

        # Pre-convert to numpy arrays for efficient indexing during training
        # Categorical features are indices (int64) for embeddings
        self.cat_data = df[self.cat_cols].values.astype(np.int64)

        # Continuous features are floats (float32)
        self.cont_data = df[self.cont_cols].values.astype(np.float32)

        # Handle targets if not in test mode
        if not self.is_test:
            if Config.TARGET_COL in df.columns:
                self.targets = df[Config.TARGET_COL].values.astype(np.float32)
            else:
                raise ValueError(
                    f"Target column '{Config.TARGET_COL}' not found in training/validation data."
                )

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.cat_data)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves the sample at the given index.

        Returns:
            dict: A dictionary containing:
                - 'cat_seq': LongTensor of shape (12,) for the Transformer branch.
                - 'cont_vec': FloatTensor of shape (n_cont,) for the Funnel branch.
                - 'target': FloatTensor of shape (1,) (only if is_test=False).
        """
        # Retrieve numpy data
        cat_seq_np = self.cat_data[idx]
        cont_vec_np = self.cont_data[idx]

        # Convert to PyTorch Tensors
        item = {
            "cat_seq": torch.tensor(cat_seq_np, dtype=torch.long),
            "cont_vec": torch.tensor(cont_vec_np, dtype=torch.float32),
        }

        # Add target if available
        if not self.is_test:
            target_val = self.targets[idx]
            # Return target as shape (1,) for BCEWithLogitsLoss compatibility
            item["target"] = torch.tensor([target_val], dtype=torch.float32)

        return item
