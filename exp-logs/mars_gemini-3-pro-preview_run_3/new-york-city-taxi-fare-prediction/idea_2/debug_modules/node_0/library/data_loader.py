import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from library.config import Config
from library.feature_engineering import process_data


class TaxiDataset(Dataset):
    """
    PyTorch Dataset for Taxi Fare Prediction.
    Separates features into categorical (for embeddings) and continuous (for dense layers).
    """

    def __init__(self, df, cat_cols, cont_cols, target_col=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): The dataframe containing features and targets.
            cat_cols (list): List of categorical column names.
            cont_cols (list): List of continuous column names.
            target_col (str): Name of the target column.
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode

        # Convert to numpy arrays for efficiency
        # Ensure categorical columns are integers for embedding lookups
        self.cat_data = df[cat_cols].values.astype(np.int64)

        # Ensure continuous columns are float32 for neural networks
        self.cont_data = df[cont_cols].values.astype(np.float32)

        if mode != "test" and target_col is not None and target_col in df.columns:
            self.targets = df[target_col].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (x_cat, x_cont, y) if target exists, else (x_cat, x_cont)
        """
        x_cat = torch.from_numpy(self.cat_data[idx])
        x_cont = torch.from_numpy(self.cont_data[idx])

        if self.targets is not None:
            y = torch.tensor(self.targets[idx])
            return x_cat, x_cont, y
        else:
            return x_cat, x_cont


def load_and_process_data(load_cached_data=True):
    """
    Orchestrates the loading and processing of Train, Validation, and Test datasets.
    Uses the feature_engineering library to handle caching and transformation.

    Args:
        load_cached_data (bool): Whether to load from parquet cache if available.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Process or Load Train Data
    train_df = process_data(mode="train", load_cached_data=load_cached_data)

    # Process or Load Validation Data
    val_df = process_data(mode="val", load_cached_data=load_cached_data)

    # Process or Load Test Data
    test_df = process_data(mode="test", load_cached_data=load_cached_data)

    return train_df, val_df, test_df


def create_dataloaders(
    train_df,
    val_df,
    test_df,
    batch_size=Config.NN_PARAMS["batch_size"],
    num_workers=Config.NN_PARAMS["num_workers"],
    load_cached_scaler=True,
):
    """
    Prepares DataLoaders for the Neural Network.
    Handles scaling of continuous features and creation of the TaxiDataset.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker processes.
        load_cached_scaler (bool): Whether to load the scaler from disk.

    Returns:
        tuple: (train_loader, val_loader, test_loader, scaler, cat_cols, cont_cols)
    """
    # 1. Define Column Groups
    # Categorical columns corresponding to embeddings in Config
    # Note: 'cluster' maps to both pickup and dropoff cluster columns
    cat_cols = [
        "pickup_cluster",
        "dropoff_cluster",
        "hour",
        "day_of_week",
        "year",
    ]

    # Continuous columns are everything else excluding metadata and target
    exclude_cols = set(
        cat_cols + [Config.ID_COL, Config.TARGET_COL, Config.DATETIME_COL]
    )
    cont_cols = [c for c in train_df.columns if c not in exclude_cols]

    # 2. Scaling
    # We must scale continuous features for the Neural Network.
    # We fit on Train and transform Val/Test.
    scaler = StandardScaler()

    scaler_path = Config.SCALER_PATH

    # Logic to load or fit scaler
    if load_cached_scaler and os.path.exists(scaler_path):
        print(f"Loading cached scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)

        # Transform all datasets
        # We use a copy to avoid modifying the original dataframe used by GBDT
        train_cont = scaler.transform(train_df[cont_cols])
        val_cont = scaler.transform(val_df[cont_cols])
        test_cont = scaler.transform(test_df[cont_cols])

    else:
        print("Fitting new scaler on training data...")
        scaler.fit(train_df[cont_cols])

        # Save scaler
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

        train_cont = scaler.transform(train_df[cont_cols])
        val_cont = scaler.transform(val_df[cont_cols])
        test_cont = scaler.transform(test_df[cont_cols])

    # Create temporary DataFrames for Dataset initialization to hold scaled values
    # We do this to keep the original DFs intact for the GBDT model which might not need scaling
    train_df_scaled = train_df.copy()
    train_df_scaled[cont_cols] = train_cont

    val_df_scaled = val_df.copy()
    val_df_scaled[cont_cols] = val_cont

    test_df_scaled = test_df.copy()
    test_df_scaled[cont_cols] = test_cont

    # 3. Create Datasets
    train_dataset = TaxiDataset(
        train_df_scaled, cat_cols, cont_cols, target_col=Config.TARGET_COL, mode="train"
    )

    val_dataset = TaxiDataset(
        val_df_scaled, cat_cols, cont_cols, target_col=Config.TARGET_COL, mode="val"
    )

    test_dataset = TaxiDataset(
        test_df_scaled, cat_cols, cont_cols, target_col=None, mode="test"
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.NN_PARAMS["pin_memory"],
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.NN_PARAMS["pin_memory"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.NN_PARAMS["pin_memory"],
    )

    return train_loader, val_loader, test_loader, scaler, cat_cols, cont_cols
