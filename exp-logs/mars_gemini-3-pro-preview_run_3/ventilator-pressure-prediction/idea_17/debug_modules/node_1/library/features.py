import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
from library.config import Config


def engineer_features(load_cached_data=True):
    """
    Orchestrates the feature engineering pipeline: loading, processing, scaling,
    reshaping, and caching data.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from disk.

    Returns:
        dict: A dictionary containing numpy arrays for train/val/test sets (x, y, u_out, ids)
              and scaler statistics.
    """
    # Define cache file paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    file_names = {
        "train_x": "train_x.npy",
        "train_y": "train_y.npy",
        "train_u_out": "train_u_out.npy",
        "val_x": "val_x.npy",
        "val_y": "val_y.npy",
        "val_u_out": "val_u_out.npy",
        "test_x": "test_x.npy",
        "test_ids": "test_ids.npy",
        "test_u_out": "test_u_out.npy",
        "scaler_stats": "scaler_stats.npz",
    }

    file_paths = {k: os.path.join(cache_dir, v) for k, v in file_names.items()}

    # 1. Attempt to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in file_paths.values())
        if all_exist:
            print(f"Loading cached features from {cache_dir}...")
            data = {}
            for k, p in file_paths.items():
                if k == "scaler_stats":
                    # Load scaler stats specially
                    stats = np.load(p)
                    data[k] = {key: stats[key] for key in stats.files}
                else:
                    data[k] = np.load(p)
            return data
        else:
            print("Cache not found or incomplete. Recomputing features...")

    # 2. Load Raw Data
    print("Loading raw datasets...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # 3. Apply Feature Engineering
    print("Applying feature engineering...")
    # We process them separately to ensure strictly no leakage, though the logic is identical.
    train_df = _compute_features(train_df)
    val_df = _compute_features(val_df)
    test_df = _compute_features(test_df)

    # 4. Prepare for Scaling and Reshaping
    # Identify feature columns (exclude IDs and Targets)
    # The _compute_features function returns the full DF with new columns.
    # We need to select the input features for the model.

    feature_cols = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "area",
        "u_in_diff",
        "u_in_next1",
        "u_in_next2",
        "u_in_next3",
        "u_in_next4",
        "u_in_diff_next1",
        "R_u_in",
        "area_C",
    ]

    # Verify columns exist
    for col in feature_cols:
        if col not in train_df.columns:
            raise ValueError(
                f"Expected feature {col} missing from processed dataframe."
            )

    # 5. Scaling
    print("Scaling features...")
    scaler = RobustScaler(quantile_range=(25.0, 75.0))

    # Fit on Train only
    scaler.fit(train_df[feature_cols])

    # Transform all sets
    # Note: We convert to float32 to save memory
    train_x_flat = scaler.transform(train_df[feature_cols]).astype(np.float32)
    val_x_flat = scaler.transform(val_df[feature_cols]).astype(np.float32)
    test_x_flat = scaler.transform(test_df[feature_cols]).astype(np.float32)

    # 6. Extract Targets and Auxiliaries
    # Targets
    train_y_flat = train_df[Config.TARGET_COL].values.astype(np.float32)
    val_y_flat = val_df[Config.TARGET_COL].values.astype(np.float32)

    # u_out (for loss masking)
    train_u_out_flat = train_df["u_out"].values.astype(np.float32)
    val_u_out_flat = val_df["u_out"].values.astype(np.float32)
    test_u_out_flat = test_df["u_out"].values.astype(np.float32)

    # IDs (for submission)
    test_ids_flat = test_df[Config.ID_COL].values.astype(np.int32)

    # 7. Reshape to (N_breaths, 80, Features)
    # We assume the data is sorted by breath_id and time_step (standard for this dataset)
    # and that every breath has exactly 80 time steps.
    BREATH_STEPS = 80

    def reshape_data(flat_data, steps=BREATH_STEPS):
        # Calculate number of breaths
        n_rows = flat_data.shape[0]
        if n_rows % steps != 0:
            raise ValueError(
                f"Data length {n_rows} is not divisible by breath length {steps}."
            )
        n_breaths = n_rows // steps

        # Handle 1D vs 2D arrays
        if flat_data.ndim == 1:
            return flat_data.reshape(n_breaths, steps)
        else:
            return flat_data.reshape(n_breaths, steps, flat_data.shape[1])

    print("Reshaping data...")
    train_x = reshape_data(train_x_flat)
    train_y = reshape_data(train_y_flat)
    train_u_out = reshape_data(train_u_out_flat)

    val_x = reshape_data(val_x_flat)
    val_y = reshape_data(val_y_flat)
    val_u_out = reshape_data(val_u_out_flat)

    test_x = reshape_data(test_x_flat)
    test_u_out = reshape_data(test_u_out_flat)
    test_ids = reshape_data(test_ids_flat)  # Shape (N, 80)

    # 8. Save to Cache
    print(f"Saving features to {cache_dir}...")
    np.save(file_paths["train_x"], train_x)
    np.save(file_paths["train_y"], train_y)
    np.save(file_paths["train_u_out"], train_u_out)

    np.save(file_paths["val_x"], val_x)
    np.save(file_paths["val_y"], val_y)
    np.save(file_paths["val_u_out"], val_u_out)

    np.save(file_paths["test_x"], test_x)
    np.save(file_paths["test_ids"], test_ids)
    np.save(file_paths["test_u_out"], test_u_out)

    # Save scaler stats
    np.savez(file_paths["scaler_stats"], center=scaler.center_, scale=scaler.scale_)

    # 9. Return
    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_u_out": train_u_out,
        "val_x": val_x,
        "val_y": val_y,
        "val_u_out": val_u_out,
        "test_x": test_x,
        "test_ids": test_ids,
        "test_u_out": test_u_out,
        "scaler_stats": {"center": scaler.center_, "scale": scaler.scale_},
    }


def _compute_features(df):
    """
    Applies the feature engineering logic defined in the Idea description.
    """
    # Ensure sorted order (critical for diff/shift)
    # Assuming data is already sorted by breath_id, id as per standard loading,
    # but explicit sort is safer if overhead is acceptable.
    # Given strict runtime, we rely on dataset structure but do vectorized checks.

    # 1. Time Delta (dt)
    # Calculate difference in time_step
    df["dt"] = df["time_step"].diff()
    # Fix boundary: The first step of a new breath should have dt=0 (or small).
    # Since we can't diff across breaths, we set dt=0 where breath_id changes.
    # Vectorized mask:
    breath_change_mask = df[Config.BREATH_COL] != df[Config.BREATH_COL].shift(1)
    df.loc[breath_change_mask, "dt"] = 0.0
    df["dt"] = df["dt"].fillna(0.0)

    # 2. Physical Integration (Area/Volume)
    # Area = Cumulative Sum of (u_in * dt) per breath
    # groupby().cumsum() is optimized
    df["area"] = (df["u_in"] * df["dt"]).groupby(df[Config.BREATH_COL]).cumsum()

    # 3. Derivatives (Acceleration)
    df["u_in_diff"] = df["u_in"].diff()
    df.loc[breath_change_mask, "u_in_diff"] = 0.0
    df["u_in_diff"] = df["u_in_diff"].fillna(0.0)

    # 4. Explicit Lookahead (t+1 to t+4)
    # We use negative shifts. We must mask out values that shift in from the next breath.
    lookahead_steps = Config.FEATURE_CONFIG.get("lookahead_steps", 4)

    for k in range(1, lookahead_steps + 1):
        col_name = f"u_in_next{k}"
        # Shift upwards (future into present)
        df[col_name] = df["u_in"].shift(-k)

        # Mask boundaries: if breath_id at i is not same as breath_id at i+k
        # checking breath_id vs breath_id shifted by -k
        boundary_mask = df[Config.BREATH_COL] != df[Config.BREATH_COL].shift(-k)
        df.loc[boundary_mask, col_name] = 0.0

        # Fill NaNs at the very end of the dataframe
        df[col_name] = df[col_name].fillna(0.0)

    # 5. Lookahead Derivative (t+1)
    if Config.FEATURE_CONFIG.get("lookahead_diff_steps", 0) > 0:
        df["u_in_diff_next1"] = df["u_in_diff"].shift(-1)
        boundary_mask = df[Config.BREATH_COL] != df[Config.BREATH_COL].shift(-1)
        df.loc[boundary_mask, "u_in_diff_next1"] = 0.0
        df["u_in_diff_next1"] = df["u_in_diff_next1"].fillna(0.0)
    else:
        # Fallback if config disabled but code expects it, though config says enabled.
        df["u_in_diff_next1"] = 0.0

    # 6. Interaction Terms
    if Config.FEATURE_CONFIG.get("use_interaction", True):
        df["R_u_in"] = df["R"] * df["u_in"]
        # Avoid division by zero if C is 0 (physically impossible here, C >= 10)
        df["area_C"] = df["area"] / df["C"]

    return df
