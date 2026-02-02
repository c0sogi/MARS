import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from library.config import Config

# Ensure reproducibility
np.random.seed(Config.SEED)


class SegregatedScaler:
    """
    Applies RobustScaler to continuous features while passing binary features raw.
    Persists scaler parameters (center, scale) to disk for consistency.
    """

    def __init__(self):
        self.continuous_features = Config.CONTINUOUS_FEATURES
        self.binary_features = Config.BINARY_FEATURES
        self.scaler = RobustScaler(quantile_range=(25.0, 75.0))
        self.fitted = False

    def fit(self, df):
        X_cont = df[self.continuous_features].values
        self.scaler.fit(X_cont)
        self.fitted = True

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transform.")

        X_cont = df[self.continuous_features].values
        X_bin = df[self.binary_features].values

        X_cont_scaled = self.scaler.transform(X_cont)

        # Concatenate: Continuous first, then Binary (matching Config.ALL_FEATURES order)
        return np.concatenate([X_cont_scaled, X_bin], axis=1)

    def save(self, path):
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted scaler.")
        np.savez(path, center=self.scaler.center_, scale=self.scaler.scale_)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found: {path}")
        data = np.load(path)
        self.scaler.center_ = data["center"]
        self.scaler.scale_ = data["scale"]
        self.fitted = True


def feature_engineering(df):
    """
    Computes time-weighted volume, physics terms, and multi-step deltas.
    """
    # Ensure sorted by breath and time
    df = df.sort_values(["breath_id", "time_step"])

    # Time delta
    # Groupby is safer than simple shift to avoid cross-breath contamination
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # Time-weighted integration: u_in * dt
    df["u_in_dt"] = df["u_in"] * df["dt"]
    df["area"] = df.groupby("breath_id")["u_in_dt"].cumsum()

    # Physics Interaction Terms
    df["R_u_in"] = df["R"] * df["u_in"]
    df["area_div_C"] = df["area"] / df["C"]

    # Multi-Step Deltas (t - (t-k))
    # u_in_diff1..2
    for k in range(1, 3):
        df[f"u_in_diff{k}"] = df.groupby("breath_id")["u_in"].diff(k).fillna(0)

    return df


class VentilatorDataset(Dataset):
    def __init__(self, X, y, u_out):
        self.X = X
        self.y = y
        self.u_out = u_out

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.X[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "u_out": torch.tensor(self.u_out[idx], dtype=torch.float32),
        }


def load_data(split, debug=False, load_cached_data=True):
    """
    Loads, processes, and caches data for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'.
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): If True, tries to load from disk cache first.

    Returns:
        VentilatorDataset: The ready-to-use PyTorch dataset.
    """
    # 1. Path Setup
    if split == "train":
        meta_path = Config.TRAIN_META
        base_cache_path = Config.CACHE_TRAIN_PATH
        raw_file = Config.TRAIN_CSV
    elif split == "val":
        meta_path = Config.VAL_META
        base_cache_path = Config.CACHE_VAL_PATH
        raw_file = Config.TRAIN_CSV
    elif split == "test":
        meta_path = Config.TEST_META
        base_cache_path = Config.CACHE_TEST_PATH
        raw_file = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown split: {split}")

    # We use .npz for the final 3D tensors (overriding .parquet config for final cache)
    cache_path = os.path.splitext(base_cache_path)[0] + ".npz"
    scaler_path = Config.CACHE_SCALER_PATH

    # Adjust paths for debug mode
    if debug:
        cache_path = cache_path.replace(".npz", "_debug.npz")
        scaler_path = scaler_path.replace(".npz", "_debug.npz")

    # 2. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} data from {cache_path}...")
        data = np.load(cache_path)
        return VentilatorDataset(data["X"], data["y"], data["u_out"])

    print(f"Processing {split} data (Debug={debug})...")

    # 3. Load Metadata
    df_meta = pd.read_csv(meta_path)
    if debug:
        # Sample breaths for debugging
        unique_breaths = df_meta["breath_id"].unique()
        sample_breaths = unique_breaths[: Config.DEBUG_SAMPLE_SIZE]
        df_meta = df_meta[df_meta["breath_id"].isin(sample_breaths)]

    target_breaths = df_meta["breath_id"].unique()

    # 4. Load Raw Data
    # Load full file and filter (efficient enough for this dataset size)
    df_raw = pd.read_csv(raw_file)
    df = df_raw[df_raw["breath_id"].isin(target_breaths)].copy()

    # 5. Feature Engineering
    df = feature_engineering(df)

    # 6. Scaling
    scaler = SegregatedScaler()

    if split == "train":
        print("Fitting scaler on training data...")
        scaler.fit(df)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        scaler.save(scaler_path)
    else:
        # For val/test, we must use the existing scaler
        if os.path.exists(scaler_path):
            print(f"Loading scaler from {scaler_path}...")
            scaler.load(scaler_path)
        else:
            if debug:
                # In debug mode, if train hasn't run, fit on current to allow standalone testing
                print("Debug: Scaler not found, fitting on current split...")
                scaler.fit(df)
            else:
                raise FileNotFoundError(
                    f"Scaler not found at {scaler_path}. Run train split first."
                )

    X_norm = scaler.transform(df)

    # 7. Reshape to (N_breaths, 80, N_features)
    seq_len = 80
    n_breaths = len(target_breaths)

    # Verify data integrity
    if len(df) != n_breaths * seq_len:
        # In case of ragged data (unlikely here), one would need padding.
        # Assuming standard dataset structure where each breath is 80 steps.
        pass

    X = X_norm.reshape(n_breaths, seq_len, -1)
    u_out = df["u_out"].values.reshape(n_breaths, seq_len)

    if "pressure" in df.columns:
        y = df["pressure"].values.reshape(n_breaths, seq_len)
    else:
        y = np.zeros((n_breaths, seq_len))

    # 8. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, X=X, y=y, u_out=u_out)
    print(f"Saved processed data to {cache_path}")

    return VentilatorDataset(X, y, u_out)
