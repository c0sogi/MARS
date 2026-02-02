import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import seed_everything


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Returns:
        x (torch.Tensor): Features of shape (80, Input_Dim)
        y (torch.Tensor): Target pressure of shape (80,)
        u_out (torch.Tensor): Control input u_out of shape (80,) for masking
    """

    def __init__(self, X, y, u_out):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.u_out[idx]


def add_features(df):
    """
    Adds PID state, physics interactions, and lookahead features.
    Uses vectorized numpy operations by reshaping to (N_breaths, 80).
    """
    # Ensure data is sorted by breath_id and time_step
    df = df.sort_values(["breath_id", "id"]).reset_index(drop=True)

    # Extract base arrays
    u_in = df["u_in"].values
    R = df["R"].values
    C = df["C"].values
    time_step = df["time_step"].values

    # Reshape to (N_breaths, 80)
    # 80 is the fixed length of breaths in this dataset
    n_breaths = len(df) // 80
    u_in_matrix = u_in.reshape(n_breaths, 80)
    time_matrix = time_step.reshape(n_breaths, 80)

    # 1. PID States
    # Calculate dt (time delta)
    # Prepend 0 to maintain shape. Assuming breath starts at t=0 or relative to previous breath end.
    # Since time_step is relative to breath start (approx 0), prepend=0 is safe.
    dt_matrix = np.diff(time_matrix, axis=1, prepend=0)

    # Integral (Volume approximation: sum(u_in * dt))
    # Cite solution_lesson_node_00001: Explicitly engineering state variables (Integrals)
    area_matrix = np.cumsum(u_in_matrix * dt_matrix, axis=1)

    # Derivative (Acceleration approximation)
    # Prepend 0 to maintain shape (diff reduces size by 1)
    u_in_diff_matrix = np.diff(u_in_matrix, axis=1, prepend=0)
    # Fix the first element (prepend puts 0 at start, diff is between t and t-1)
    # The diff at t=0 is u_in[0] - 0 (assuming prev state 0) or just 0.
    # np.diff with prepend=0 calculates val[0]-0, val[1]-val[0]. This is correct.

    # 2. Lookahead Features (Shift Left)
    # u_in at t+1
    u_in_next1_matrix = np.roll(u_in_matrix, -1, axis=1)
    u_in_next1_matrix[:, -1] = 0  # Zero out the wrap-around

    # u_in at t+2
    u_in_next2_matrix = np.roll(u_in_matrix, -2, axis=1)
    u_in_next2_matrix[:, -1] = 0
    u_in_next2_matrix[:, -2] = 0

    # u_in_diff at t+1
    u_in_diff_next1_matrix = np.roll(u_in_diff_matrix, -1, axis=1)
    u_in_diff_next1_matrix[:, -1] = 0

    # Flatten back to 1D
    df["area"] = area_matrix.flatten()
    df["dt"] = dt_matrix.flatten()
    df["u_in_diff"] = u_in_diff_matrix.flatten()
    df["u_in_next1"] = u_in_next1_matrix.flatten()
    df["u_in_next2"] = u_in_next2_matrix.flatten()
    df["u_in_diff_next1"] = u_in_diff_next1_matrix.flatten()

    # 3. Physics Interactions
    df["R_u_in"] = df["R"] * df["u_in"]
    df["vol_C"] = df["area"] / df["C"]

    return df


def prepare_datasets(load_cached_data=True, debug=Config.DEBUG):
    """
    Main function to load, process, and return datasets.
    Handles caching of processed numpy arrays.
    """
    seed_everything(Config.SEED)

    # Define cache paths
    cache_files = [
        Config.TRAIN_CACHE_X,
        Config.TRAIN_CACHE_Y,
        Config.TRAIN_CACHE_U_OUT,
        Config.VAL_CACHE_X,
        Config.VAL_CACHE_Y,
        Config.VAL_CACHE_U_OUT,
        Config.TEST_CACHE_X,
        Config.TEST_CACHE_IDS,
        Config.TEST_CACHE_U_OUT,
    ]

    # Check if cache exists
    cache_exists = all(os.path.exists(f) for f in cache_files)

    if load_cached_data and cache_exists:
        print("Loading cached data...")
        train_x = np.load(Config.TRAIN_CACHE_X)
        train_y = np.load(Config.TRAIN_CACHE_Y)
        train_u_out = np.load(Config.TRAIN_CACHE_U_OUT)

        val_x = np.load(Config.VAL_CACHE_X)
        val_y = np.load(Config.VAL_CACHE_Y)
        val_u_out = np.load(Config.VAL_CACHE_U_OUT)

        test_x = np.load(Config.TEST_CACHE_X)
        # test_ids = np.load(Config.TEST_CACHE_IDS) # Not needed for Dataset object, but loaded if needed later
        test_u_out = np.load(Config.TEST_CACHE_U_OUT)

        # Create dummy targets for test set (all zeros)
        test_y = np.zeros((test_x.shape[0], 80), dtype=np.float32)

    else:
        print("Processing data from scratch...")
        # Load Raw Data
        train_df = pd.read_csv(Config.TRAIN_CSV)
        val_df = pd.read_csv(Config.VAL_CSV)
        test_df = pd.read_csv(Config.TEST_CSV)

        # Add dummy pressure to test for consistent processing
        test_df["pressure"] = 0

        if debug:
            print("DEBUG MODE: Subsampling data...")
            # Take first 100 breaths
            train_df = train_df.iloc[: 100 * 80]
            val_df = val_df.iloc[: 100 * 80]
            test_df = test_df.iloc[: 100 * 80]

        # Feature Engineering
        print("Generating features...")
        train_df = add_features(train_df)
        val_df = add_features(val_df)
        test_df = add_features(test_df)

        # Scaling
        print("Fitting RobustScaler...")
        scaler = RobustScaler()
        # Fit only on training data
        scaler.fit(train_df[Config.FEATURE_COLS])

        # Save scaler statistics
        np.savez(Config.SCALER_PATH, center=scaler.center_, scale=scaler.scale_)

        # Transform all sets
        train_df[Config.FEATURE_COLS] = scaler.transform(train_df[Config.FEATURE_COLS])
        val_df[Config.FEATURE_COLS] = scaler.transform(val_df[Config.FEATURE_COLS])
        test_df[Config.FEATURE_COLS] = scaler.transform(test_df[Config.FEATURE_COLS])

        # Reshape to (N_breaths, 80, Features)
        print("Reshaping data...")

        def reshape_data(df):
            n_breaths = len(df) // 80
            x = df[Config.FEATURE_COLS].values.reshape(n_breaths, 80, -1)
            y = df["pressure"].values.reshape(n_breaths, 80)
            u_out = df["u_out"].values.reshape(n_breaths, 80)
            ids = df["id"].values.reshape(n_breaths, 80)
            return x, y, u_out, ids

        train_x, train_y, train_u_out, _ = reshape_data(train_df)
        val_x, val_y, val_u_out, _ = reshape_data(val_df)
        test_x, test_y, test_u_out, test_ids = reshape_data(test_df)

        # Save to Cache
        print("Saving to cache...")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(Config.TRAIN_CACHE_X, train_x)
        np.save(Config.TRAIN_CACHE_Y, train_y)
        np.save(Config.TRAIN_CACHE_U_OUT, train_u_out)

        np.save(Config.VAL_CACHE_X, val_x)
        np.save(Config.VAL_CACHE_Y, val_y)
        np.save(Config.VAL_CACHE_U_OUT, val_u_out)

        np.save(Config.TEST_CACHE_X, test_x)
        np.save(Config.TEST_CACHE_IDS, test_ids)
        np.save(Config.TEST_CACHE_U_OUT, test_u_out)

    # Create Datasets
    train_dataset = VentilatorDataset(train_x, train_y, train_u_out)
    val_dataset = VentilatorDataset(val_x, val_y, val_u_out)
    test_dataset = VentilatorDataset(
        test_x, test_y, test_u_out
    )  # test_y is dummy zeros

    print(
        f"Data Loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset
