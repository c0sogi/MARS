import os
import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from library import config

# Mappings for categorical features to integer indices for embeddings
R_MAP = {5: 0, 20: 1, 50: 2}
C_MAP = {10: 0, 20: 1, 50: 2}


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes physics-based features derived from the Equation of Motion.
    Also adds temporal derivative features (lags and diffs).
    """
    # Calculate dt
    df["dt"] = df.groupby(config.BREATH_ID_COL)[config.TIME_COL].diff().fillna(0)

    # Calculate cumulative volume: integral of u_in * dt
    df["volume"] = (df["u_in"] * df["dt"]).groupby(df[config.BREATH_ID_COL]).cumsum()

    # Physics interactions (Cite solution_lesson_node_00005)
    df["flow_interaction"] = df["u_in"] * df["R"]
    df["volume_interaction"] = df["volume"] / df["C"]
    df["cumulative_volume"] = df["volume"]

    # Lag features (Cite solution_lesson_node_00022)
    for lag in range(1, 5):
        df[f"u_in_lag{lag}"] = (
            df.groupby(config.BREATH_ID_COL)["u_in"].shift(lag).fillna(0)
        )

    for lag in range(1, 3):
        df[f"u_out_lag{lag}"] = (
            df.groupby(config.BREATH_ID_COL)["u_out"].shift(lag).fillna(0)
        )

    # Diff features (Cite solution_lesson_node_00022)
    for diff in range(1, 5):
        df[f"u_in_diff{diff}"] = (
            df.groupby(config.BREATH_ID_COL)["u_in"].diff(diff).fillna(0)
        )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deprecated. R and C are treated as continuous.
    """
    return df


def get_cache_hash():
    """
    Generates a hash based on the feature configuration to ensure cache validity.
    """
    feature_str = (
        f"{config.CONTINUOUS_FEATURES}_{config.CATEGORICAL_FEATURES}_"
        f"{config.PHYSICS_FEATURES}_{config.SEQ_LEN}"
    )
    return hashlib.md5(feature_str.encode()).hexdigest()


def save_scaler(scaler, path_center, path_scale):
    np.save(path_center, scaler.center_)
    np.save(path_scale, scaler.scale_)


def load_scaler(path_center, path_scale):
    scaler = RobustScaler()
    scaler.center_ = np.load(path_center)
    scaler.scale_ = np.load(path_scale)
    return scaler


def process_dataframe(df, scaler=None, is_train=False):
    """
    Applies feature engineering, scaling, and reshaping.
    """
    # 1. Physics Features
    df = add_physics_features(df)

    # 2. Categorical Encoding (Removed)

    # 3. Scaling
    features_to_scale = config.CONTINUOUS_FEATURES

    if is_train:
        scaler = RobustScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for validation/test sets")
        df[features_to_scale] = scaler.transform(df[features_to_scale])

    # 4. Reshaping to (N_breaths, 80, N_features)
    num_breaths = len(df) // config.SEQ_LEN

    # Extract arrays
    x_cont = df[config.CONTINUOUS_FEATURES].values.reshape(
        num_breaths, config.SEQ_LEN, len(config.CONTINUOUS_FEATURES)
    )

    # Targets
    if config.TARGET_COL in df.columns:
        y = df[config.TARGET_COL].values.reshape(num_breaths, config.SEQ_LEN)
    else:
        y = None

    # Meta info
    ids = df[config.ID_COL].values.reshape(num_breaths, config.SEQ_LEN)

    return x_cont, None, None, y, ids, scaler


def prepare_datasets(load_cached_data=True):
    """
    Main entry point to load, process, and return datasets.
    Handles caching logic.
    """
    cache_hash = get_cache_hash()
    cache_dir = config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define filenames
    files = {
        "train_x_cont": os.path.join(cache_dir, f"train_x_cont_{cache_hash}.npy"),
        "train_x_cat": os.path.join(cache_dir, f"train_x_cat_{cache_hash}.npy"),
        "train_x_phys": os.path.join(cache_dir, f"train_x_phys_{cache_hash}.npy"),
        "train_y": os.path.join(cache_dir, f"train_y_{cache_hash}.npy"),
        "train_uout": os.path.join(cache_dir, f"train_uout_{cache_hash}.npy"),
        "train_ids": os.path.join(cache_dir, f"train_ids_{cache_hash}.npy"),
        "val_x_cont": os.path.join(cache_dir, f"val_x_cont_{cache_hash}.npy"),
        "val_x_cat": os.path.join(cache_dir, f"val_x_cat_{cache_hash}.npy"),
        "val_x_phys": os.path.join(cache_dir, f"val_x_phys_{cache_hash}.npy"),
        "val_y": os.path.join(cache_dir, f"val_y_{cache_hash}.npy"),
        "val_uout": os.path.join(cache_dir, f"val_uout_{cache_hash}.npy"),
        "val_ids": os.path.join(cache_dir, f"val_ids_{cache_hash}.npy"),
        "test_x_cont": os.path.join(cache_dir, f"test_x_cont_{cache_hash}.npy"),
        "test_x_cat": os.path.join(cache_dir, f"test_x_cat_{cache_hash}.npy"),
        "test_x_phys": os.path.join(cache_dir, f"test_x_phys_{cache_hash}.npy"),
        "test_ids": os.path.join(cache_dir, f"test_ids_{cache_hash}.npy"),
        "test_uout": os.path.join(cache_dir, f"test_uout_{cache_hash}.npy"),
        "scaler_center": os.path.join(cache_dir, f"scaler_center_{cache_hash}.npy"),
        "scaler_scale": os.path.join(cache_dir, f"scaler_scale_{cache_hash}.npy"),
    }

    # Check if all files exist
    all_exist = all(os.path.exists(f) for f in files.values())

    if load_cached_data and all_exist:
        print(f"Loading cached data from {cache_dir}...")
        data = {}
        for k, v in files.items():
            if "scaler" not in k:
                data[k] = np.load(v)
        return data

    print("Cache missing or reload requested. Processing data from scratch...")

    # Load Raw Data
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # Process Train
    train_uout_raw = train_df["u_out"].values.reshape(-1, config.SEQ_LEN)
    train_df = add_physics_features(train_df)

    scaler = RobustScaler()
    train_df[config.CONTINUOUS_FEATURES] = scaler.fit_transform(
        train_df[config.CONTINUOUS_FEATURES]
    )
    save_scaler(scaler, files["scaler_center"], files["scaler_scale"])

    num_train = len(train_df) // config.SEQ_LEN
    train_x_cont = train_df[config.CONTINUOUS_FEATURES].values.reshape(
        num_train, config.SEQ_LEN, -1
    )
    train_y = train_df[config.TARGET_COL].values.reshape(num_train, config.SEQ_LEN)
    train_ids = train_df[config.ID_COL].values.reshape(num_train, config.SEQ_LEN)

    # Process Val
    val_uout_raw = val_df["u_out"].values.reshape(-1, config.SEQ_LEN)
    val_df = add_physics_features(val_df)
    val_df[config.CONTINUOUS_FEATURES] = scaler.transform(
        val_df[config.CONTINUOUS_FEATURES]
    )
    num_val = len(val_df) // config.SEQ_LEN
    val_x_cont = val_df[config.CONTINUOUS_FEATURES].values.reshape(
        num_val, config.SEQ_LEN, -1
    )
    val_y = val_df[config.TARGET_COL].values.reshape(num_val, config.SEQ_LEN)
    val_ids = val_df[config.ID_COL].values.reshape(num_val, config.SEQ_LEN)

    # Process Test
    test_uout_raw = test_df["u_out"].values.reshape(-1, config.SEQ_LEN)
    test_df = add_physics_features(test_df)
    test_df[config.CONTINUOUS_FEATURES] = scaler.transform(
        test_df[config.CONTINUOUS_FEATURES]
    )
    num_test = len(test_df) // config.SEQ_LEN
    test_x_cont = test_df[config.CONTINUOUS_FEATURES].values.reshape(
        num_test, config.SEQ_LEN, -1
    )
    test_ids = test_df[config.ID_COL].values.reshape(num_test, config.SEQ_LEN)

    # Save to cache
    np.save(files["train_x_cont"], train_x_cont)
    np.save(files["train_y"], train_y)
    np.save(files["train_uout"], train_uout_raw)
    np.save(files["train_ids"], train_ids)

    np.save(files["val_x_cont"], val_x_cont)
    np.save(files["val_y"], val_y)
    np.save(files["val_uout"], val_uout_raw)
    np.save(files["val_ids"], val_ids)

    np.save(files["test_x_cont"], test_x_cont)
    np.save(files["test_ids"], test_ids)
    np.save(files["test_uout"], test_uout_raw)

    print("Data processing complete and cached.")

    return {
        "train_x_cont": train_x_cont,
        "train_y": train_y,
        "train_uout": train_uout_raw,
        "train_ids": train_ids,
        "val_x_cont": val_x_cont,
        "val_y": val_y,
        "val_uout": val_uout_raw,
        "val_ids": val_ids,
        "test_x_cont": test_x_cont,
        "test_ids": test_ids,
        "test_uout": test_uout_raw,
    }
