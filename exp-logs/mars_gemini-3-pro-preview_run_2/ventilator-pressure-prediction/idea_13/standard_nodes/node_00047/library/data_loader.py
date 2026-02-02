import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    """

    def __init__(self, features, targets=None, u_outs=None, is_test=False):
        """
        Args:
            features (np.ndarray): Input features of shape (N, Seq_Len, Feature_Dim).
            targets (np.ndarray, optional): Target pressure values of shape (N, Seq_Len).
            u_outs (np.ndarray, optional): Control input u_out of shape (N, Seq_Len).
            is_test (bool): Flag indicating if this is the test set.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.is_test = is_test

        if not self.is_test:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

        # u_out is needed for loss weighting (train) or potentially for post-processing/analysis
        if u_outs is not None:
            self.u_outs = torch.tensor(u_outs, dtype=torch.float32)
        else:
            self.u_outs = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]

        if self.is_test:
            # For test, we might still return u_out if needed, but primarily features
            # Returning dummy target 0 for consistency if needed by caller,
            # but usually inference loop handles this.
            # Returning: features, dummy_target, u_out
            u_out = (
                self.u_outs[idx] if self.u_outs is not None else torch.zeros(x.shape[0])
            )
            return x, torch.zeros(x.shape[0]), u_out

        y = self.targets[idx]
        u_out = self.u_outs[idx]
        return x, y, u_out


def engineer_features(df):
    """
    Performs feature engineering on the dataframe.
    Calculates time-weighted volume, physics interaction terms, lags, and diffs.
    """
    # Ensure sorted by breath_id and time_step
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # Identify start of new breaths
    df["breath_id_lag"] = df["breath_id"].shift(1)
    df["breath_id_lag"] = df["breath_id_lag"].fillna(0)
    same_breath = df["breath_id"] == df["breath_id_lag"]

    # 1. Time Delta (dt)
    df["dt"] = df["time_step"].diff()
    df.loc[~same_breath, "dt"] = 0
    df["dt"] = df["dt"].fillna(0)

    # 2. Time-Weighted Volume (Integration)
    # volume = cumsum(u_in * dt)
    df["area"] = df["u_in"] * df["dt"]
    df["u_in_cumsum"] = df.groupby("breath_id")["area"].cumsum()

    # 3. Physics Interaction Terms
    # Resistive Pressure ~ R * u_in
    df["R_u_in"] = df["R"] * df["u_in"]
    # Elastic Pressure ~ Volume / C
    df["vol_C"] = df["u_in_cumsum"] / df["C"]

    # 4. Lag Features
    for i in range(1, 3):
        col_name = f"u_in_lag{i}"
        df[col_name] = df["u_in"].shift(i)
        df.loc[~same_breath, col_name] = 0
        df[col_name] = df[col_name].fillna(0)

    # 5. Finite Differences
    # First derivative
    df["u_in_diff1"] = df["u_in"].diff()
    df.loc[~same_breath, "u_in_diff1"] = 0
    df["u_in_diff1"] = df["u_in_diff1"].fillna(0)

    # Second derivative
    df["u_in_diff2"] = df["u_in_diff1"].diff()
    df.loc[~same_breath, "u_in_diff2"] = 0
    df["u_in_diff2"] = df["u_in_diff2"].fillna(0)

    # Cleanup auxiliary columns
    df.drop(["breath_id_lag", "area"], axis=1, inplace=True)

    return df


def process_data(debug=False):
    """
    Loads raw data, performs splitting based on metadata, engineers features,
    scales data, and saves processed files to cache.
    """
    print("Loading raw data...")
    train_raw = pd.read_csv(Config.TRAIN_CSV)
    test_raw = pd.read_csv(Config.TEST_CSV)

    train_meta = pd.read_csv(Config.TRAIN_METADATA)
    val_meta = pd.read_csv(Config.VAL_METADATA)

    # Get Breath IDs
    train_breath_ids = set(train_meta["breath_id"].unique())
    val_breath_ids = set(val_meta["breath_id"].unique())

    if debug:
        print("DEBUG Mode: Subsampling data...")
        train_breath_ids = set(list(train_breath_ids)[:100])
        val_breath_ids = set(list(val_breath_ids)[:50])
        # For test, we just take the first few breaths
        test_breath_ids = set(test_raw["breath_id"].unique()[:50])
        test_raw = test_raw[test_raw["breath_id"].isin(test_breath_ids)].copy()

    # Split Train.csv into Train and Val
    print("Splitting train/val...")
    df_train = train_raw[train_raw["breath_id"].isin(train_breath_ids)].copy()
    df_val = train_raw[train_raw["breath_id"].isin(val_breath_ids)].copy()
    df_test = test_raw.copy()

    del train_raw  # Free memory

    # Engineer Features
    print("Engineering features...")
    df_train = engineer_features(df_train)
    df_val = engineer_features(df_val)
    df_test = engineer_features(df_test)

    # Define features to scale
    # u_out is binary, do not scale
    # id, breath_id, pressure are not input features
    feature_cols = [
        "time_step",
        "u_in",
        "R",
        "C",
        "u_in_cumsum",
        "R_u_in",
        "vol_C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff1",
        "u_in_diff2",
    ]

    # Scaling
    if Config.USE_ROBUST_SCALER:
        print("Fitting RobustScaler...")
        scaler = RobustScaler()
        scaler.fit(df_train[feature_cols])

        print("Transforming data...")
        df_train[feature_cols] = scaler.transform(df_train[feature_cols])
        df_val[feature_cols] = scaler.transform(df_val[feature_cols])
        df_test[feature_cols] = scaler.transform(df_test[feature_cols])

        # Save Scaler Params
        np.savez(Config.SCALER_CACHE, center=scaler.center_, scale=scaler.scale_)

    # Save processed dataframes
    print("Saving processed data to cache...")
    df_train.to_parquet(Config.TRAIN_CACHE, index=False)
    df_val.to_parquet(Config.VAL_CACHE, index=False)
    df_test.to_parquet(Config.TEST_CACHE, index=False)

    return df_train, df_val, df_test


def get_dataloaders(debug=Config.DEBUG, load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders.
    Handles caching logic.
    """
    # Check if cache exists
    cache_exists = (
        os.path.exists(Config.TRAIN_CACHE)
        and os.path.exists(Config.VAL_CACHE)
        and os.path.exists(Config.TEST_CACHE)
        and os.path.exists(Config.SCALER_CACHE)
    )

    if load_cached_data and cache_exists:
        print("Loading cached processed data...")
        df_train = pd.read_parquet(Config.TRAIN_CACHE)
        df_val = pd.read_parquet(Config.VAL_CACHE)
        df_test = pd.read_parquet(Config.TEST_CACHE)

        if debug:
            # If debug is requested but we loaded full cache, subsample here
            print("Subsampling cached data for DEBUG...")
            train_b_ids = df_train["breath_id"].unique()[:100]
            val_b_ids = df_val["breath_id"].unique()[:50]
            test_b_ids = df_test["breath_id"].unique()[:50]

            df_train = df_train[df_train["breath_id"].isin(train_b_ids)]
            df_val = df_val[df_val["breath_id"].isin(val_b_ids)]
            df_test = df_test[df_test["breath_id"].isin(test_b_ids)]

    else:
        print("Processing data from scratch...")
        df_train, df_val, df_test = process_data(debug=debug)

    # Prepare features list
    # Note: u_out is treated separately in Dataset, but usually included in input features as well
    # The model architecture likely expects u_out as a feature.
    # Based on "Idea", we use u_out in loss weighting.
    # We will include u_out in the feature matrix AND return it separately for loss.
    feature_cols = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "u_in_cumsum",
        "R_u_in",
        "vol_C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff1",
        "u_in_diff2",
    ]

    target_col = "pressure"
    u_out_col = "u_out"

    def reshape_to_sequences(df, is_test=False):
        # Assuming fixed sequence length of 80
        # If not 80, this reshape will fail.
        # The dataset is standard 80 steps per breath.
        # Ensure sorted
        df = df.sort_values(["breath_id", "time_step"])

        num_breaths = len(df) // 80
        if len(df) % 80 != 0:
            raise ValueError(
                f"Data length {len(df)} is not divisible by 80. Reshape failed."
            )

        features = df[feature_cols].values.reshape(num_breaths, 80, len(feature_cols))
        u_outs = df[u_out_col].values.reshape(num_breaths, 80)

        if not is_test:
            targets = df[target_col].values.reshape(num_breaths, 80)
            return features, targets, u_outs
        else:
            return features, None, u_outs

    print("Reshaping data for LSTM...")
    x_train, y_train, u_out_train = reshape_to_sequences(df_train, is_test=False)
    x_val, y_val, u_out_val = reshape_to_sequences(df_val, is_test=False)
    x_test, _, u_out_test = reshape_to_sequences(df_test, is_test=True)

    print(f"Train shape: {x_train.shape}")
    print(f"Val shape: {x_val.shape}")
    print(f"Test shape: {x_test.shape}")

    # Create Datasets
    train_dataset = VentilatorDataset(x_train, y_train, u_out_train, is_test=False)
    val_dataset = VentilatorDataset(x_val, y_val, u_out_val, is_test=False)
    test_dataset = VentilatorDataset(x_test, None, u_out_test, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
