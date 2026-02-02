import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.data_utils import preprocess_pipeline


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control Data.

    Handles the storage and retrieval of:
    - Integer-encoded categorical features (for Embeddings)
    - Normalized continuous features (for MLP)
    - Binary targets (for Classification)
    """

    def __init__(self, df, cat_cols, cont_cols, target_col=None):
        """
        Args:
            df (pd.DataFrame): The dataframe containing features and optionally targets.
            cat_cols (list): List of categorical column names.
            cont_cols (list): List of continuous column names.
            target_col (str, optional): Name of the target column. Defaults to None.
        """
        # Ensure data is in the correct format for PyTorch
        # Categorical features -> Int64 (Long) for Embedding lookup
        self.cat_features = df[cat_cols].values.astype(np.int64)

        # Continuous features -> Float32 for Linear layers
        self.cont_features = df[cont_cols].values.astype(np.float32)

        # Target -> Float32 for BCEWithLogitsLoss
        if target_col and target_col in df.columns:
            self.targets = df[target_col].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.cat_features)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the features and target for a given index.
        """
        item = {
            "cat_features": torch.tensor(self.cat_features[idx], dtype=torch.long),
            "cont_features": torch.tensor(self.cont_features[idx], dtype=torch.float32),
        }

        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, load_cached_data=True, max_samples=None
):
    """
    Orchestrates the data pipeline and returns PyTorch DataLoaders.

    Args:
        batch_size (int): Batch size for the dataloaders.
        load_cached_data (bool): Whether to load processed data from cache.
        max_samples (int, optional): If set, limits the dataset size for debugging.

    Returns:
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
        test_loader (DataLoader): DataLoader for test set.
        metadata (dict): Metadata dictionary containing vocab sizes and column names.
    """
    # Delegate heavy processing and caching to the provided utility
    train_df, val_df, test_df, metadata = preprocess_pipeline(
        load_cached_data=load_cached_data
    )

    # Optional subsampling for debugging
    if max_samples is not None and max_samples > 0:
        train_df = train_df.iloc[:max_samples]
        val_df = val_df.iloc[:max_samples]
        test_df = test_df.iloc[:max_samples]

    cat_cols = metadata["cat_cols"]
    cont_cols = metadata["cont_cols"]

    # Instantiate Datasets
    train_dataset = ManufacturingDataset(
        train_df, cat_cols, cont_cols, target_col="target"
    )
    val_dataset = ManufacturingDataset(val_df, cat_cols, cont_cols, target_col="target")
    test_dataset = ManufacturingDataset(test_df, cat_cols, cont_cols, target_col=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader, metadata
