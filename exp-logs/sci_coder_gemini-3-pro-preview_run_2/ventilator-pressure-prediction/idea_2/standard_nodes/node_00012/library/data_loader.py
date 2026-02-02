import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config


class VentilatorDataset(Dataset):
    def __init__(self, cont_features, cat_features, targets=None, ids=None):
        """
        Args:
            cont_features (np.array): Continuous features of shape (N, Seq_Len, F_cont)
            cat_features (np.array): Categorical features of shape (N, Seq_Len, F_cat)
            targets (np.array, optional): Targets of shape (N, Seq_Len)
            ids (np.array, optional): IDs of shape (N, Seq_Len)
        """
        self.cont_features = torch.tensor(cont_features, dtype=torch.float32)
        self.cat_features = torch.tensor(cat_features, dtype=torch.long)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )
        self.ids = torch.tensor(ids, dtype=torch.long) if ids is not None else None

    def __len__(self):
        return len(self.cont_features)

    def __getitem__(self, idx):
        data = {"cont": self.cont_features[idx], "cat": self.cat_features[idx]}
        if self.targets is not None:
            data["target"] = self.targets[idx]
        if self.ids is not None:
            data["ids"] = self.ids[idx]
        return data


def add_physics_features(df):
    """
    Computes physics-based and lag features for the ventilator dataset.
    """
    # Ensure data is sorted by breath_id and time_step/id
    df = df.sort_values(["breath_id", "id"]).reset_index(drop=True)

    # --- Physics Integrations ---
    # Cumulative sum of u_in (approximation of volume)
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    # --- Interactions ---
    # Resistive Pressure Proxy: Flow * Resistance
    df["R_u_in"] = df["R"] * df["u_in"]
    # Elastic Pressure Proxy: Volume / Compliance
    df["u_in_cumsum_div_C"] = df["u_in_cumsum"] / df["C"]

    # --- Dynamics (Lags & Diffs) ---
    # We use groupby shift to ensure we don't shift across different breaths
    grp = df.groupby("breath_id")

    df["u_in_lag1"] = grp["u_in"].shift(1).fillna(0)
    df["u_in_lag2"] = grp["u_in"].shift(2).fillna(0)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_diff1"] - (df["u_in_lag1"] - df["u_in_lag2"])

    # --- Categorical Mapping ---
    # Map R: {5, 20, 50} -> {0, 1, 2}
    r_map = {5: 0, 20: 1, 50: 2}
    df["R_cat"] = df["R"].map(r_map)

    # Map C: {10, 20, 50} -> {0, 1, 2}
    c_map = {10: 0, 20: 1, 50: 2}
    df["C_cat"] = df["C"].map(c_map)

    return df


def preprocess_data(load_cached_data=True, debug=False):
    """
    Loads, processes, scales, and reshapes data. Handles caching.
    """
    # Define cache paths
    train_cache = Config.TRAIN_CACHE
    val_cache = Config.VAL_CACHE
    test_cache = Config.TEST_CACHE
    scaler_cache = Config.SCALER_CACHE

    # Check if we can load from cache
    caches_exist = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(scaler_cache)
    )

    if load_cached_data and caches_exist:
        print("Loading data from cache...")
        train_data = np.load(train_cache)
        val_data = np.load(val_cache)
        test_data = np.load(test_cache)

        # Return dictionaries of arrays
        return (
            {k: train_data[k] for k in train_data},
            {k: val_data[k] for k in val_data},
            {k: test_data[k] for k in test_data},
        )

    print("Processing data from scratch...")

    # 1. Load Raw Data
    df_train_raw = pd.read_csv(Config.TRAIN_CSV)
    df_test_raw = pd.read_csv(Config.TEST_CSV)

    # Load Metadata for splitting
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA)
    df_val_meta = pd.read_csv(Config.VAL_METADATA)

    train_breath_ids = set(df_train_meta["breath_id"].unique())
    val_breath_ids = set(df_val_meta["breath_id"].unique())

    if debug:
        # Reduce dataset size for debugging
        train_breath_ids = set(list(train_breath_ids)[: Config.DEBUG_SAMPLES])
        val_breath_ids = set(list(val_breath_ids)[: Config.DEBUG_SAMPLES])
        df_train_raw = df_train_raw[
            df_train_raw["breath_id"].isin(train_breath_ids | val_breath_ids)
        ]
        # Take a subset of test as well
        test_breath_ids = df_test_raw["breath_id"].unique()[: Config.DEBUG_SAMPLES]
        df_test_raw = df_test_raw[df_test_raw["breath_id"].isin(test_breath_ids)]

    # 2. Feature Engineering
    print("Generating physics features...")
    df_train_full = add_physics_features(df_train_raw)
    df_test = add_physics_features(df_test_raw)

    # 3. Split Train/Val
    print("Splitting train and validation sets...")
    df_train = df_train_full[df_train_full["breath_id"].isin(train_breath_ids)].copy()
    df_val = df_train_full[df_train_full["breath_id"].isin(val_breath_ids)].copy()

    # Ensure sorting
    df_train = df_train.sort_values(["breath_id", "id"])
    df_val = df_val.sort_values(["breath_id", "id"])
    df_test = df_test.sort_values(["breath_id", "id"])

    # 4. Scaling
    print("Scaling continuous features...")
    scaler = RobustScaler()

    # Fit only on training data
    scaler.fit(df_train[Config.CONT_FEATURES])

    # Transform all sets
    df_train[Config.CONT_FEATURES] = scaler.transform(df_train[Config.CONT_FEATURES])
    df_val[Config.CONT_FEATURES] = scaler.transform(df_val[Config.CONT_FEATURES])
    df_test[Config.CONT_FEATURES] = scaler.transform(df_test[Config.CONT_FEATURES])

    # Save scaler params (center and scale)
    np.savez(scaler_cache, center=scaler.center_, scale=scaler.scale_)

    # 5. Reshape and Extract
    def extract_arrays(df, has_target=True):
        num_breaths = len(df) // Config.SEQ_LEN

        # Continuous features
        cont = df[Config.CONT_FEATURES].values.reshape(num_breaths, Config.SEQ_LEN, -1)

        # Categorical features (R_cat, C_cat)
        # Stack them: (N, 80, 2)
        r_cat = df["R_cat"].values.reshape(num_breaths, Config.SEQ_LEN, 1)
        c_cat = df["C_cat"].values.reshape(num_breaths, Config.SEQ_LEN, 1)
        cat = np.concatenate([r_cat, c_cat], axis=2)

        # IDs (for submission/tracking)
        ids = df["id"].values.reshape(num_breaths, Config.SEQ_LEN)

        result = {"cont": cont, "cat": cat, "ids": ids}

        if has_target:
            targets = df[Config.TARGET_COL].values.reshape(num_breaths, Config.SEQ_LEN)
            result["targets"] = targets

        return result

    print("Reshaping arrays...")
    train_arrays = extract_arrays(df_train, has_target=True)
    val_arrays = extract_arrays(df_val, has_target=True)
    test_arrays = extract_arrays(df_test, has_target=False)

    # 6. Save to Cache
    print(f"Saving to {Config.WORKING_DIR}...")
    np.savez(train_cache, **train_arrays)
    np.savez(val_cache, **val_arrays)
    np.savez(test_cache, **test_arrays)

    return train_arrays, val_arrays, test_arrays


def get_data_loaders(load_cached_data=True, debug=False):
    """
    Returns train, val, and test DataLoaders.
    """
    train_data, val_data, test_data = preprocess_data(load_cached_data, debug)

    # Create Datasets
    train_dataset = VentilatorDataset(
        train_data["cont"],
        train_data["cat"],
        train_data["targets"],
        ids=train_data.get("ids"),
    )

    val_dataset = VentilatorDataset(
        val_data["cont"],
        val_data["cat"],
        val_data["targets"],
        ids=val_data.get("ids"),
    )

    test_dataset = VentilatorDataset(
        test_data["cont"], test_data["cat"], targets=None, ids=test_data.get("ids")
    )

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
