import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from library.config import (
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    ID_COL,
    CACHE_DIR,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    NUM_WORKERS,
    DEBUG,
)
from library.feature_engineering import (
    get_processed_data,
    fit_scaler,
    transform_data,
    save_scaler,
    load_scaler,
)


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Stores data in 3D tensors: (Num_Breaths, Seq_Len, Num_Features).
    """

    def __init__(self, X, y=None, u_out=None, ids=None):
        """
        Args:
            X (np.ndarray): Input features of shape (N, Seq_Len, F).
            y (np.ndarray, optional): Target pressure of shape (N, Seq_Len).
            u_out (np.ndarray, optional): Expiratory valve status of shape (N, Seq_Len).
            ids (np.ndarray, optional): Time step IDs of shape (N, Seq_Len).
        """
        self.X = torch.tensor(X, dtype=torch.float32)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

        if u_out is not None:
            self.u_out = torch.tensor(u_out, dtype=torch.float32)
        else:
            self.u_out = None

        if ids is not None:
            self.ids = torch.tensor(ids, dtype=torch.long)
        else:
            self.ids = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"x": self.X[idx]}

        if self.y is not None:
            item["y"] = self.y[idx]

        if self.u_out is not None:
            item["u_out"] = self.u_out[idx]

        if self.ids is not None:
            item["ids"] = self.ids[idx]

        return item


def _reshape_to_sequences(df, feature_cols, target_col=None, id_col=None):
    """
    Reshapes a flat dataframe into (Num_Breaths, Seq_Len, Features).
    Assumes dataframe is sorted by breath_id and time_step, and each breath has exactly MAX_SEQ_LEN steps.
    """
    num_rows = len(df)
    if num_rows % MAX_SEQ_LEN != 0:
        # In case of debug/subsampling where breaths might be cut, or data irregularities
        # We truncate to the nearest full breath for safety, though data should be clean
        num_breaths = num_rows // MAX_SEQ_LEN
        num_rows = num_breaths * MAX_SEQ_LEN
        df = df.iloc[:num_rows]
    else:
        num_breaths = num_rows // MAX_SEQ_LEN

    # Extract Features
    # Ensure columns are in a fixed order
    X_flat = df[feature_cols].values.astype(np.float32)
    X = X_flat.reshape(num_breaths, MAX_SEQ_LEN, len(feature_cols))

    # Extract Targets
    y = None
    if target_col and target_col in df.columns:
        y_flat = df[target_col].values.astype(np.float32)
        y = y_flat.reshape(num_breaths, MAX_SEQ_LEN)

    # Extract u_out (specifically for auxiliary usage in metric/loss)
    u_out = None
    if "u_out" in df.columns:
        u_out_flat = df["u_out"].values.astype(np.float32)
        u_out = u_out_flat.reshape(num_breaths, MAX_SEQ_LEN)

    # Extract IDs
    ids = None
    if id_col and id_col in df.columns:
        ids_flat = df[id_col].values.astype(np.int64)
        ids = ids_flat.reshape(num_breaths, MAX_SEQ_LEN)

    return X, y, u_out, ids


def prepare_data_loaders(
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, debug=DEBUG, load_cached_data=True
):
    """
    Prepares DataLoaders for train, validation, and test sets.
    Handles caching of reshaped tensors to avoid re-processing overhead.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    debug_suffix = "_debug" if debug else ""

    # Define cache paths for the tensors
    cache_files = {
        "train": os.path.join(CACHE_DIR, f"train_tensors{debug_suffix}.npz"),
        "val": os.path.join(CACHE_DIR, f"val_tensors{debug_suffix}.npz"),
        "test": os.path.join(CACHE_DIR, f"test_tensors{debug_suffix}.npz"),
    }

    # Check if all caches exist
    all_caches_exist = all(os.path.exists(p) for p in cache_files.values())

    # --- 1. Load from Tensor Cache if available ---
    if load_cached_data and all_caches_exist:
        print("Loading tensor data from cache...")
        data_arrays = {}
        for split, path in cache_files.items():
            loaded = np.load(path)
            data_arrays[split] = {
                "X": loaded["X"],
                "y": loaded["y"] if "y" in loaded else None,
                "u_out": loaded["u_out"] if "u_out" in loaded else None,
                "ids": loaded["ids"] if "ids" in loaded else None,
            }
            print(f"Loaded {split} tensors: {data_arrays[split]['X'].shape}")

    else:
        # --- 2. Process from DataFrames ---
        print("Processing data from DataFrames (Cache miss or force reload)...")

        # Load processed DataFrames (feature engineering applied)
        df_train = get_processed_data(
            "train", debug=debug, load_cached_data=load_cached_data
        )
        df_val = get_processed_data(
            "val", debug=debug, load_cached_data=load_cached_data
        )
        df_test = get_processed_data(
            "test", debug=debug, load_cached_data=load_cached_data
        )

        # Fit or Load Scaler
        # We always fit on train, unless we are in a pure inference mode where train might not be loaded.
        # Here we assume standard training pipeline.
        scaler = fit_scaler(df_train)
        save_scaler(scaler, debug=debug)

        # Transform Data
        print("Scaling data...")
        df_train = transform_data(df_train, scaler)
        df_val = transform_data(df_val, scaler)
        df_test = transform_data(df_test, scaler)

        # Define feature columns
        feature_cols = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

        data_arrays = {}

        # Reshape and Cache Train
        print("Reshaping and caching Train data...")
        X_train, y_train, u_out_train, ids_train = _reshape_to_sequences(
            df_train, feature_cols, TARGET_COL, ID_COL
        )
        np.savez(
            cache_files["train"], X=X_train, y=y_train, u_out=u_out_train, ids=ids_train
        )
        data_arrays["train"] = {
            "X": X_train,
            "y": y_train,
            "u_out": u_out_train,
            "ids": ids_train,
        }

        # Reshape and Cache Val
        print("Reshaping and caching Val data...")
        X_val, y_val, u_out_val, ids_val = _reshape_to_sequences(
            df_val, feature_cols, TARGET_COL, ID_COL
        )
        np.savez(cache_files["val"], X=X_val, y=y_val, u_out=u_out_val, ids=ids_val)
        data_arrays["val"] = {
            "X": X_val,
            "y": y_val,
            "u_out": u_out_val,
            "ids": ids_val,
        }

        # Reshape and Cache Test
        print("Reshaping and caching Test data...")
        X_test, y_test, u_out_test, ids_test = _reshape_to_sequences(
            df_test, feature_cols, None, ID_COL
        )
        np.savez(
            cache_files["test"], X=X_test, u_out=u_out_test, ids=ids_test
        )  # No y for test
        data_arrays["test"] = {
            "X": X_test,
            "y": None,
            "u_out": u_out_test,
            "ids": ids_test,
        }

    # --- 3. Create Datasets and DataLoaders ---
    print("Creating DataLoaders...")

    train_dataset = VentilatorDataset(
        X=data_arrays["train"]["X"],
        y=data_arrays["train"]["y"],
        u_out=data_arrays["train"]["u_out"],
        ids=data_arrays["train"]["ids"],
    )

    val_dataset = VentilatorDataset(
        X=data_arrays["val"]["X"],
        y=data_arrays["val"]["y"],
        u_out=data_arrays["val"]["u_out"],
        ids=data_arrays["val"]["ids"],
    )

    test_dataset = VentilatorDataset(
        X=data_arrays["test"]["X"],
        y=None,
        u_out=data_arrays["test"]["u_out"],
        ids=data_arrays["test"]["ids"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
