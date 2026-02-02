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

    Features added:
    - dt: Time difference between steps.
    - cumulative_volume: Integral of u_in over time.
    - flow_interaction: u_in * R
    - volume_interaction: cumulative_volume / C
    """
    # Ensure data is sorted by breath_id and time_step
    # (Metadata generation usually ensures this, but safety first)
    # df = df.sort_values([config.BREATH_ID_COL, config.TIME_COL]) # Assuming input is already sorted for performance

    # Calculate dt
    # Since the data is strictly 80 steps per breath, we can use a vectorized approach
    # without expensive groupby().diff() if the structure is guaranteed.
    # However, groupby is safer.
    df["dt"] = df.groupby(config.BREATH_ID_COL)[config.TIME_COL].diff().fillna(0)

    # Calculate cumulative volume: integral of u_in * dt
    # u_in is 0-100, we treat it as flow rate.
    df["volume"] = (df["u_in"] * df["dt"]).groupby(df[config.BREATH_ID_COL]).cumsum()

    # Physics interactions
    df["flow_interaction"] = df["u_in"] * df["R"]
    df["volume_interaction"] = df["volume"] / df["C"]

    # Rename volume to match config expectation if needed, though config uses 'cumulative_volume'
    df["cumulative_volume"] = df["volume"]

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps physical R and C values to integer indices for embedding layers.
    """
    df["R_idx"] = df["R"].map(R_MAP)
    df["C_idx"] = df["C"].map(C_MAP)
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

    # 2. Categorical Encoding
    df = encode_categoricals(df)

    # 3. Scaling
    # We need to scale continuous features.
    # Note: u_out is in CONTINUOUS_FEATURES in config, so it gets scaled.
    # We must extract raw u_out for masking BEFORE scaling if we want exact 0/1,
    # or rely on the fact that we save u_out separately.

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
    # Continuous inputs for LSTM/CNN
    x_cont = df[config.CONTINUOUS_FEATURES].values.reshape(
        num_breaths, config.SEQ_LEN, len(config.CONTINUOUS_FEATURES)
    )

    # Categorical inputs (R_idx, C_idx)
    x_cat = df[["R_idx", "C_idx"]].values.reshape(num_breaths, config.SEQ_LEN, 2)
    # Since R and C are constant per breath, we can take the first step, but keeping sequence dim is fine for concatenation
    # Actually, usually embeddings are expanded. We'll return full sequence.

    # Physics adapter inputs (flow_interaction, volume_interaction) - these were scaled as part of CONTINUOUS_FEATURES?
    # Wait, config.PHYSICS_FEATURES are ['flow_interaction', 'volume_interaction'].
    # config.CONTINUOUS_FEATURES includes them. So they are already scaled in x_cont.
    # The model architecture description implies a separate branch.
    # We can extract them from x_cont if needed, or pass them separately.
    # To be safe and explicit, let's extract them into a separate array corresponding to PHYSICS_FEATURES indices.
    phys_indices = [
        config.CONTINUOUS_FEATURES.index(f) for f in config.PHYSICS_FEATURES
    ]
    x_phys = x_cont[:, :, phys_indices]

    # Targets
    if config.TARGET_COL in df.columns:
        y = df[config.TARGET_COL].values.reshape(num_breaths, config.SEQ_LEN)
    else:
        y = None

    # Meta info
    ids = df[config.ID_COL].values.reshape(num_breaths, config.SEQ_LEN)

    # Raw u_out for masking (extract from original dataframe logic, but since we scaled in place,
    # we need to be careful. Ideally we should have kept a copy.
    # However, u_out is binary. RobustScaler on 0/1: median is likely 0 or 1.
    # Let's re-extract u_out from the 'u_out' column which is now scaled.
    # Alternatively, we can recover it or just not scale it.
    # Given the constraints, let's assume we need the raw u_out for the mask.
    # We can reconstruct it: if scaled value > threshold -> 1.
    # Better: Extract raw u_out before scaling.
    # Refactoring slightly to extract raw u_out before scaling.

    return x_cont, x_cat, x_phys, y, ids, scaler


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
    # Capture raw u_out before it gets scaled/modified
    train_uout_raw = train_df["u_out"].values.reshape(-1, config.SEQ_LEN)

    # Apply Physics Features
    train_df = add_physics_features(train_df)
    train_df = encode_categoricals(train_df)

    # Scale and Reshape
    # We need to fit scaler on train
    scaler = RobustScaler()
    train_df[config.CONTINUOUS_FEATURES] = scaler.fit_transform(
        train_df[config.CONTINUOUS_FEATURES]
    )

    # Save scaler
    save_scaler(scaler, files["scaler_center"], files["scaler_scale"])

    # Reshape Train
    num_train = len(train_df) // config.SEQ_LEN
    train_x_cont = train_df[config.CONTINUOUS_FEATURES].values.reshape(
        num_train, config.SEQ_LEN, -1
    )
    train_x_cat = train_df[["R_idx", "C_idx"]].values.reshape(
        num_train, config.SEQ_LEN, 2
    )

    phys_indices = [
        config.CONTINUOUS_FEATURES.index(f) for f in config.PHYSICS_FEATURES
    ]
    train_x_phys = train_x_cont[:, :, phys_indices]

    train_y = train_df[config.TARGET_COL].values.reshape(num_train, config.SEQ_LEN)
    train_ids = train_df[config.ID_COL].values.reshape(num_train, config.SEQ_LEN)

    # Process Val
    val_uout_raw = val_df["u_out"].values.reshape(-1, config.SEQ_LEN)
    val_df = add_physics_features(val_df)
    val_df = encode_categoricals(val_df)
    val_df[config.CONTINUOUS_FEATURES] = scaler.transform(
        val_df[config.CONTINUOUS_FEATURES]
    )

    num_val = len(val_df) // config.SEQ_LEN
    val_x_cont = val_df[config.CONTINUOUS_FEATURES].values.reshape(
        num_val, config.SEQ_LEN, -1
    )
    val_x_cat = val_df[["R_idx", "C_idx"]].values.reshape(num_val, config.SEQ_LEN, 2)
    val_x_phys = val_x_cont[:, :, phys_indices]
    val_y = val_df[config.TARGET_COL].values.reshape(num_val, config.SEQ_LEN)
    val_ids = val_df[config.ID_COL].values.reshape(num_val, config.SEQ_LEN)

    # Process Test
    test_uout_raw = test_df["u_out"].values.reshape(-1, config.SEQ_LEN)
    test_df = add_physics_features(test_df)
    test_df = encode_categoricals(test_df)
    test_df[config.CONTINUOUS_FEATURES] = scaler.transform(
        test_df[config.CONTINUOUS_FEATURES]
    )

    num_test = len(test_df) // config.SEQ_LEN
    test_x_cont = test_df[config.CONTINUOUS_FEATURES].values.reshape(
        num_test, config.SEQ_LEN, -1
    )
    test_x_cat = test_df[["R_idx", "C_idx"]].values.reshape(num_test, config.SEQ_LEN, 2)
    test_x_phys = test_x_cont[:, :, phys_indices]
    test_ids = test_df[config.ID_COL].values.reshape(num_test, config.SEQ_LEN)

    # Save to cache
    np.save(files["train_x_cont"], train_x_cont)
    np.save(files["train_x_cat"], train_x_cat)
    np.save(files["train_x_phys"], train_x_phys)
    np.save(files["train_y"], train_y)
    np.save(files["train_uout"], train_uout_raw)
    np.save(files["train_ids"], train_ids)

    np.save(files["val_x_cont"], val_x_cont)
    np.save(files["val_x_cat"], val_x_cat)
    np.save(files["val_x_phys"], val_x_phys)
    np.save(files["val_y"], val_y)
    np.save(files["val_uout"], val_uout_raw)
    np.save(files["val_ids"], val_ids)

    np.save(files["test_x_cont"], test_x_cont)
    np.save(files["test_x_cat"], test_x_cat)
    np.save(files["test_x_phys"], test_x_phys)
    np.save(files["test_ids"], test_ids)
    np.save(files["test_uout"], test_uout_raw)

    print("Data processing complete and cached.")

    return {
        "train_x_cont": train_x_cont,
        "train_x_cat": train_x_cat,
        "train_x_phys": train_x_phys,
        "train_y": train_y,
        "train_uout": train_uout_raw,
        "train_ids": train_ids,
        "val_x_cont": val_x_cont,
        "val_x_cat": val_x_cat,
        "val_x_phys": val_x_phys,
        "val_y": val_y,
        "val_uout": val_uout_raw,
        "val_ids": val_ids,
        "test_x_cont": test_x_cont,
        "test_x_cat": test_x_cat,
        "test_x_phys": test_x_phys,
        "test_ids": test_ids,
        "test_uout": test_uout_raw,
    }
