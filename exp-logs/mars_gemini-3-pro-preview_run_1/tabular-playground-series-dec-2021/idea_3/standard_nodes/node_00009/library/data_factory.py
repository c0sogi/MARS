import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


class DataFactory:
    """
    Handles data ingestion, feature engineering, and preprocessing for the Hybrid Stacking Strategy.
    """

    @staticmethod
    def load_and_engineer_data(load_cached_data=True):
        """
        Loads the train, val, and test data, applies physics-informed feature engineering,
        and manages caching using Parquet files.

        Args:
            load_cached_data (bool): If True, attempts to load processed data from disk.

        Returns:
            tuple: (train_df, val_df, test_df, test_ids)
        """
        seed_everything(Config.SEED)

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Cache paths
        train_cache = Config.TRAIN_PROCESSED_PATH
        val_cache = Config.VAL_PROCESSED_PATH
        test_cache = Config.TEST_PROCESSED_PATH

        # Check if cache exists
        if (
            load_cached_data
            and os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print(f"Loading processed data from cache: {Config.WORKING_DIR}")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)

            # Handle Test IDs
            # We assume the cached test_df includes the Id column to preserve it,
            # but we separate it for the return value to ensure features are clean.
            if Config.ID_COL in test_df.columns:
                test_ids = test_df[Config.ID_COL].values
                test_df = test_df.drop(columns=[Config.ID_COL])
            else:
                # Fallback: read raw to get IDs if missing in cache
                raw_test = pd.read_csv(Config.TEST_PATH)
                test_ids = raw_test[Config.ID_COL].values

            return train_df, val_df, test_df, test_ids

        print("Processing data from scratch...")

        # Load raw data
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Extract Test IDs
        test_ids = test_df[Config.ID_COL].values

        # Drop Id from Train/Val (Test Id handled later)
        if Config.ID_COL in train_df.columns:
            train_df = train_df.drop(columns=[Config.ID_COL])
        if Config.ID_COL in val_df.columns:
            val_df = val_df.drop(columns=[Config.ID_COL])

        # Feature Engineering Function
        def engineer_features(df):
            # 1. Euclidean distance to hydrology: sqrt(H^2 + V^2)
            if (
                "Horizontal_Distance_To_Hydrology" in df.columns
                and "Vertical_Distance_To_Hydrology" in df.columns
            ):
                df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
                    df["Horizontal_Distance_To_Hydrology"] ** 2
                    + df["Vertical_Distance_To_Hydrology"] ** 2
                )

            # 2. Relative Elevation: Elevation - Vertical_Distance_To_Hydrology
            if (
                "Elevation" in df.columns
                and "Vertical_Distance_To_Hydrology" in df.columns
            ):
                df["Relative_Elevation"] = (
                    df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
                )

            # 3. Cyclic Aspect
            if "Aspect" in df.columns:
                # Convert to radians
                aspect_rad = np.radians(df["Aspect"])
                df["Aspect_Sin"] = np.sin(aspect_rad)
                df["Aspect_Cos"] = np.cos(aspect_rad)
                # Drop original Aspect to reduce noise/collinearity
                df = df.drop(columns=["Aspect"])

            return df

        print("Applying feature engineering...")
        train_df = engineer_features(train_df)
        val_df = engineer_features(val_df)
        test_df = engineer_features(test_df)  # Test still has Id column here

        # Save to cache
        # We save test_df WITH Id so we can recover it later if needed
        print("Saving processed data to cache...")
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

        # Drop Id from test_df for return
        if Config.ID_COL in test_df.columns:
            test_df = test_df.drop(columns=[Config.ID_COL])

        return train_df, val_df, test_df, test_ids

    @staticmethod
    def get_xgb_data(load_cached_data=True):
        """
        Prepares data for XGBoost training.

        Returns:
            dict: {
                "train": (X_train, y_train),
                "val": (X_val, y_val),
                "test": X_test,
                "test_ids": test_ids
            }
        """
        train_df, val_df, test_df, test_ids = DataFactory.load_and_engineer_data(
            load_cached_data
        )

        target = Config.TARGET_COL

        # Prepare Train
        X_train = train_df.drop(columns=[target])
        # Map 1-7 to 0-6
        y_train = train_df[target] - 1

        # Prepare Val
        X_val = val_df.drop(columns=[target])
        y_val = val_df[target] - 1

        # Prepare Test (already clean)
        X_test = test_df

        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": X_test,
            "test_ids": test_ids,
        }

    @staticmethod
    def create_nn_datasets(load_cached_data=True):
        """
        Prepares PyTorch TensorDatasets for the Neural Network.
        Applies StandardScaler to features.

        Returns:
            tuple: (train_dataset, val_dataset, test_dataset, input_dim)
        """
        train_df, val_df, test_df, _ = DataFactory.load_and_engineer_data(
            load_cached_data
        )
        target = Config.TARGET_COL

        # Separate features and targets
        X_train_np = train_df.drop(columns=[target]).values.astype(np.float32)
        y_train_np = (train_df[target] - 1).values.astype(np.int64)

        X_val_np = val_df.drop(columns=[target]).values.astype(np.float32)
        y_val_np = (val_df[target] - 1).values.astype(np.int64)

        X_test_np = test_df.values.astype(np.float32)

        # Scaling
        print("Scaling features for Neural Network...")
        scaler = StandardScaler()
        # Fit on Train, transform all
        X_train_scaled = scaler.fit_transform(X_train_np)
        X_val_scaled = scaler.transform(X_val_np)
        X_test_scaled = scaler.transform(X_test_np)

        # Create TensorDatasets
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_scaled), torch.from_numpy(y_train_np)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_scaled), torch.from_numpy(y_val_np)
        )
        test_dataset = TensorDataset(torch.from_numpy(X_test_scaled))

        input_dim = X_train_scaled.shape[1]

        return train_dataset, val_dataset, test_dataset, input_dim
