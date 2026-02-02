import os
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from joblib import Parallel, delayed
from library.config import Config


def calculate_entropy(x):
    """Calculates Shannon entropy of the normalized absolute distribution."""
    if len(x) == 0:
        return 0.0
    # Normalize to treat as probabilities
    p = np.abs(x)
    s = np.sum(p)
    if s == 0:
        return 0.0
    p = p / s
    # Compute entropy
    return scipy.stats.entropy(p)


def compute_kinematic_features(signal, sensor_name, params):
    """
    Computes features based on Savitzky-Golay derivatives (Trend, Velocity, Acceleration).
    Cite {solution_lesson_node_00059}: Granular quantiles on Trend and Derivatives.
    """
    feats = {}
    quantiles = params["quantiles"]

    # Parameters for SG filter
    window = params["sg_window_length"]
    poly = params["sg_polyorder"]

    # Iterate through requested derivatives
    for deriv in params["sg_derivs"]:
        # Apply filter
        try:
            deriv_sig = scipy.signal.savgol_filter(
                signal, window_length=window, polyorder=poly, deriv=deriv
            )
        except Exception:
            deriv_sig = np.zeros_like(signal)

        if deriv == 0:
            suffix = "trend"
        elif deriv == 1:
            suffix = "vel"
        elif deriv == 2:
            suffix = "acc"
        else:
            suffix = f"deriv{deriv}"

        # Dense Quantiles
        q_vals = np.quantile(deriv_sig, quantiles)
        for q, val in zip(quantiles, q_vals):
            feats[f"{sensor_name}_{suffix}_q{int(q*100):02d}"] = val

        # Range of motion
        feats[f"{sensor_name}_{suffix}_range"] = np.max(deriv_sig) - np.min(deriv_sig)

    return feats


def compute_texture_features(signal, sensor_name, params):
    """
    Computes robust statistical moments on the Texture (Residual) signal.
    Cite {solution_lesson_node_00049}: Moments on Texture.
    """
    feats = {}

    # Robust amplitude metrics
    feats[f"{sensor_name}_txt_rms"] = np.sqrt(np.mean(signal**2))
    feats[f"{sensor_name}_txt_mad"] = np.mean(np.abs(signal - np.mean(signal)))

    # Shape metrics
    feats[f"{sensor_name}_txt_skew"] = scipy.stats.skew(signal)
    feats[f"{sensor_name}_txt_kurt"] = scipy.stats.kurtosis(signal)

    return feats


def compute_spectral_features(signal, sensor_name, params):
    """
    Computes Raw Intensity stats, PSD Band Power, and Aggregated Windowed RMS.
    Cite {solution_lesson_node_00050}: Shift Invariance via Aggregation.
    """
    feats = {}
    fs = params["sampling_rate"]

    # --- 1. Raw Intensity ---
    feats[f"{sensor_name}_raw_min"] = np.min(signal)
    feats[f"{sensor_name}_raw_max"] = np.max(signal)
    feats[f"{sensor_name}_raw_p2p"] = np.max(signal) - np.min(signal)

    # --- 2. PSD Band Power ---
    # Welch's method
    freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=min(len(signal), 256))

    for low, high in params["band_freqs"]:
        idx = np.logical_and(freqs >= low, freqs <= high)
        if np.sum(idx) > 0:
            band_power = np.mean(psd[idx])
        else:
            band_power = 0.0
        feats[f"{sensor_name}_psd_{low}-{high}Hz"] = band_power

    # --- 3. Windowed Statistics (Aggregated) ---
    n_windows = params["n_windows"]
    windows = np.array_split(signal, n_windows)

    rms_list = []
    for w_data in windows:
        if len(w_data) > 0:
            rms_list.append(np.sqrt(np.mean(w_data**2)))
        else:
            rms_list.append(0.0)

    # Aggregate to preserve shift invariance
    feats[f"{sensor_name}_win_rms_mean"] = np.mean(rms_list)
    feats[f"{sensor_name}_win_rms_std"] = np.std(rms_list)
    feats[f"{sensor_name}_win_rms_min"] = np.min(rms_list)
    feats[f"{sensor_name}_win_rms_max"] = np.max(rms_list)

    return feats


def process_segment(row, params):
    """
    Processes a single data segment (file).
    """
    segment_id = int(row["segment_id"])
    file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

    # Load Data
    try:
        # Using float32 for memory efficiency
        df = pd.read_csv(file_path, dtype="float32")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Imputation: Fill NaNs with column mean (per segment)
    df = df.fillna(df.mean()).fillna(0)

    # Initialize feature dict
    features = {"segment_id": segment_id}
    if "time_to_eruption" in row:
        features["time_to_eruption"] = row["time_to_eruption"]

    # Process each sensor
    # Sensors are named sensor_1 to sensor_10
    for i in range(1, params["n_sensors"] + 1):
        sensor_col = f"sensor_{i}"
        if sensor_col not in df.columns:
            continue

        signal = df[sensor_col].values

        # Apply the Dual-Transform Strategy

        # 1. Kinematic Features (Trend, Vel, Acc)
        # Cite {solution_lesson_node_00049}: Kinematics on Trend
        k_feats = compute_kinematic_features(signal, sensor_col, params)
        features.update(k_feats)

        # 2. Texture Features (Residuals)
        # Cite {solution_lesson_node_00049}: Decomposition
        trend = scipy.signal.savgol_filter(
            signal,
            window_length=params["sg_window_length"],
            polyorder=params["sg_polyorder"],
        )
        texture = signal - trend

        t_feats = compute_texture_features(texture, sensor_col, params)
        features.update(t_feats)

        # 3. Spectral/Intensity Features (Raw)
        s_feats = compute_spectral_features(signal, sensor_col, params)
        features.update(s_feats)

    return features


def generate_features(
    metadata_path, output_path, load_cached_data=True, debug_limit=None
):
    """
    Main driver to generate features for a dataset defined by metadata.
    Handles caching and parallel processing.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached features from {output_path}...")
        return pd.read_parquet(output_path)

    print(f"Generating features from {metadata_path}...")

    # 2. Load Metadata
    meta_df = pd.read_csv(metadata_path)

    if debug_limit:
        print(f"DEBUG: Limiting to {debug_limit} samples.")
        meta_df = meta_df.head(debug_limit)

    # 3. Parallel Processing
    # Convert DataFrame rows to list of dicts for iteration
    rows = meta_df.to_dict("records")

    params = Config.FEATURE_PARAMS

    results = Parallel(n_jobs=Config.N_CORES, verbose=0)(
        delayed(process_segment)(row, params) for row in rows
    )

    # Filter out Nones (failed reads)
    results = [r for r in results if r is not None]

    # 4. Aggregate
    feature_df = pd.DataFrame(results)

    # Ensure segment_id is int
    if "segment_id" in feature_df.columns:
        feature_df["segment_id"] = feature_df["segment_id"].astype(int)

    # 5. Save to Cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feature_df.to_parquet(output_path, index=False)
    print(f"Features saved to {output_path}. Shape: {feature_df.shape}")

    return feature_df


def run_feature_engineering(load_cached_data=True, debug_limit=None):
    """
    Orchestrates the feature generation for Train, Val, and Test sets.
    """
    # Train
    print("\n--- Processing Training Set ---")
    train_df = generate_features(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_FEATURES_PATH,
        load_cached_data=load_cached_data,
        debug_limit=debug_limit,
    )

    # Val
    print("\n--- Processing Validation Set ---")
    val_df = generate_features(
        Config.VAL_METADATA_PATH,
        Config.VAL_FEATURES_PATH,
        load_cached_data=load_cached_data,
        debug_limit=debug_limit,
    )

    # Test
    print("\n--- Processing Test Set ---")
    test_df = generate_features(
        Config.TEST_METADATA_PATH,
        Config.TEST_FEATURES_PATH,
        load_cached_data=load_cached_data,
        debug_limit=debug_limit,
    )

    return train_df, val_df, test_df
