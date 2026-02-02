import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.features import engineer_features


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.

    Attributes:
        x (torch.Tensor): Dynamic input features (N_breaths, Seq_Len, Input_Dim).
        static (torch.Tensor): Static lung attributes (N_breaths, 2).
        u_out (torch.Tensor): Binary control flag for loss masking (N_breaths, Seq_Len).
        y (torch.Tensor, optional): Target pressure values (N_breaths, Seq_Len).
    """

    def __init__(self, x, static, u_out, y=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.static = torch.tensor(static, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        item = {"x": self.x[idx], "static": self.static[idx], "u_out": self.u_out[idx]}
        if self.y is not None:
            item["y"] = self.y[idx]
        return item


def _get_feature_indices():
    """Helper to find indices of R and C in the feature list."""
    # The input X is formed by CONT_FEATURES + BINARY_FEATURES
    feature_list = Config.CONT_FEATURES + Config.BINARY_FEATURES
    try:
        r_idx = feature_list.index(Config.R_COL)
        c_idx = feature_list.index(Config.C_COL)
    except ValueError:
        raise ValueError("R or C column not found in feature list.")
    return r_idx, c_idx


def _process_and_cache(dataset_type, load_cached_data=True):
    """
    Loads engineered features, reshapes them into tensors, and caches the result.

    Args:
        dataset_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from .npy cache.

    Returns:
        tuple: (x, static, u_out, y, ids)
            - x: (N, 80, Input_Dim)
            - static: (N, 2)
            - u_out: (N, 80)
            - y: (N, 80) or None
            - ids: (N, 80) or None (only for test)
    """
    cache_dir = os.path.join(Config.WORKING_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    prefix = f"{dataset_type}"
    if Config.DEBUG:
        prefix += "_debug"

    path_x = os.path.join(cache_dir, f"{prefix}_x.npy")
    path_static = os.path.join(cache_dir, f"{prefix}_static.npy")
    path_uout = os.path.join(cache_dir, f"{prefix}_uout.npy")
    path_y = os.path.join(cache_dir, f"{prefix}_y.npy")
    path_ids = os.path.join(cache_dir, f"{prefix}_ids.npy")

    # Try loading from cache
    if load_cached_data:
        try:
            if (
                os.path.exists(path_x)
                and os.path.exists(path_static)
                and os.path.exists(path_uout)
            ):
                # Check optional files depending on mode
                if dataset_type == "test" and not os.path.exists(path_ids):
                    raise FileNotFoundError
                if dataset_type != "test" and not os.path.exists(path_y):
                    raise FileNotFoundError

                print(f"Loading reshaped {dataset_type} tensors from cache...")
                x = np.load(path_x)
                static = np.load(path_static)
                u_out = np.load(path_uout)
                y = np.load(path_y) if dataset_type != "test" else None
                ids = np.load(path_ids) if dataset_type == "test" else None
                return x, static, u_out, y, ids
        except Exception as e:
            print(f"Cache load failed for {dataset_type}: {e}. Recomputing...")

    # Compute from scratch
    # 1. Load engineered dataframe
    df = engineer_features(dataset_type, load_cached_data=load_cached_data)

    if Config.DEBUG:
        # Take a subset of breaths
        unique_breaths = df[Config.BREATH_ID_COL].unique()
        subset_breaths = unique_breaths[:100]
        df = df[df[Config.BREATH_ID_COL].isin(subset_breaths)].copy()
        print(f"DEBUG: Subsampled {dataset_type} to {len(subset_breaths)} breaths.")

    # 2. Verify shape
    if len(df) % Config.SEQ_LEN != 0:
        raise ValueError(
            f"Data length {len(df)} is not divisible by SEQ_LEN {Config.SEQ_LEN}"
        )

    n_breaths = len(df) // Config.SEQ_LEN

    # 3. Extract Features
    # Input X: Continuous + Binary
    feature_cols = Config.CONT_FEATURES + Config.BINARY_FEATURES
    x_flat = df[feature_cols].values.astype(np.float32)
    x = x_flat.reshape(n_breaths, Config.SEQ_LEN, -1)

    # 4. Extract Static Features (R, C)
    # They are part of x, we just need to slice them out.
    # Since they are constant per breath, we take the value at time_step=0 for each breath
    r_idx, c_idx = _get_feature_indices()
    # x is (N, 80, F), we want (N, 2)
    # We take the first time step [:, 0, :] and then the specific columns
    static = x[:, 0, [r_idx, c_idx]]

    # 5. Extract u_out for masking
    u_out_col = Config.U_OUT_COL
    # u_out is in BINARY_FEATURES, which are at the end of feature_cols
    u_out_idx = feature_cols.index(u_out_col)
    u_out = x[:, :, u_out_idx]  # (N, 80)

    # 6. Extract Targets (y) or IDs
    y = None
    ids = None

    if dataset_type != "test":
        y_flat = df[Config.TARGET_COL].values.astype(np.float32)
        y = y_flat.reshape(n_breaths, Config.SEQ_LEN)
        np.save(path_y, y)
    else:
        ids_flat = df[Config.ID_COL].values.astype(np.int64)
        # We don't necessarily need IDs reshaped as (N, 80) for the model,
        # but for reconstruction it helps. We'll save flattened or reshaped.
        # Let's save flattened IDs for submission mapping, but here we return whatever is needed.
        # Actually, returning flattened IDs is usually easier for submission.
        ids = ids_flat
        np.save(path_ids, ids)

    # 7. Save to cache
    np.save(path_x, x)
    np.save(path_static, static)
    np.save(path_uout, u_out)

    return x, static, u_out, y, ids


def prepare_datasets(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Prepares DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for training/inference.
        num_workers (int): Number of workers for DataLoaders.
        load_cached_data (bool): Whether to use cached intermediate files.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    # Train
    x_train, static_train, u_out_train, y_train, _ = _process_and_cache(
        "train", load_cached_data
    )
    train_dataset = VentilatorDataset(x_train, static_train, u_out_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Val
    x_val, static_val, u_out_val, y_val, _ = _process_and_cache("val", load_cached_data)
    val_dataset = VentilatorDataset(x_val, static_val, u_out_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test
    x_test, static_test, u_out_test, _, test_ids = _process_and_cache(
        "test", load_cached_data
    )
    test_dataset = VentilatorDataset(x_test, static_test, u_out_test, None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
