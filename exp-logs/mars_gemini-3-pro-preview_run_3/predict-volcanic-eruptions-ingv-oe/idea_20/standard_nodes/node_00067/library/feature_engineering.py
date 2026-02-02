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
    Computes features based on Savitzky-Golay derivatives (Velocity, Acceleration).
    """
    feats = {}
    quantiles = params["quantiles"]

    # Parameters for SG filter
    window = params["sg_window_length"]
    poly = params["sg_polyorder"]

    # Iterate through requested derivatives (e.g., 1=Velocity, 2=Acceleration)
    for deriv in params["sg_derivs"]:
        # Apply filter
        try:
            deriv_sig = scipy.signal.savgol_filter(
                signal, window_length=window, polyorder=poly, deriv=deriv
            )
        except Exception:
            # Fallback if signal is too short (unlikely given 60k rows)
            deriv_sig = np.zeros_like(signal)

        suffix = "vel" if deriv == 1 else "acc" if deriv == 2 else f"deriv{deriv}"

        # Dense Quantiles
        q_vals = np.quantile(deriv_sig, quantiles)
        for q, val in zip(quantiles, q_vals):
            feats[f"{sensor_name}_{suffix}_q{int(q*100):02d}"] = val

        # Range of motion
        feats[f"{sensor_name}_{suffix}_range"] = np.max(deriv_sig) - np.min(deriv_sig)

    return feats


def compute_texture_features(signal, sensor_name, params):
    """
    Computes features on the high-frequency residual (Texture).
    Cite solution_lesson_node_00049: Decompose Time-Series into Trend and Texture.
    """
    feats = {}

    # Robust statistics on Texture
    # Cite solution_lesson_node_00021: Moments (Skewness, Kurtosis) are critical.
    if len(signal) > 0:
        feats[f"{sensor_name}_texture_rms"] = np.sqrt(np.mean(signal**2))
        feats[f"{sensor_name}_texture_skew"] = scipy.stats.skew(signal)
        feats[f"{sensor_name}_texture_kurt"] = scipy.stats.kurtosis(signal)

        # Cite solution_lesson_node_00031: Include raw extrema.
        feats[f"{sensor_name}_texture_max"] = np.max(signal)
        feats[f"{sensor_name}_texture_min"] = np.min(signal)
    else:
        feats[f"{sensor_name}_texture_rms"] = 0.0
        feats[f"{sensor_name}_texture_skew"] = 0.0
        feats[f"{sensor_name}_texture_kurt"] = 0.0
        feats[f"{sensor_name}_texture_max"] = 0.0
        feats[f"{sensor_name}_texture_min"] = 0.0

    return feats


def compute_spectral_features(signal, sensor_name, params):
    """
    Computes Raw Intensity stats, PSD Band Power, and Windowed RMS.
    """
    feats = {}
    fs = params["sampling_rate"]

    # --- 1. Raw Intensity ---
    feats[f"{sensor_name}_raw_min"] = np.min(signal)
    feats[f"{sensor_name}_raw_max"] = np.max(signal)
    feats[f"{sensor_name}_raw_p2p"] = np.max(signal) - np.min(signal)

    # --- 2. PSD Band Power ---
    # Welch's method (Cite solution_lesson_node_00054: Prefer Welch over Periodogram)
    freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=min(len(signal), 256))

    for low, high in params["band_freqs"]:
        # Find indices
        idx = np.logical_and(freqs >= low, freqs <= high)
        # Integrate power
        if np.sum(idx) > 0:
            band_power = np.mean(psd[idx])
        else:
            band_power = 0.0
        feats[f"{sensor_name}_psd_{low}-{high}Hz"] = band_power

    # --- 3. Windowed Statistics (Aggregated) ---
    # Cite solution_lesson_node_00050: Shift Invariance vs. Temporal Specificity
    # Aggregate sub-window statistics instead of flattening to reduce overfitting.
    n_windows = params["n_windows"]
    windows = np.array_split(signal, n_windows)

    w_rms = []
    w_mean = []

    for w_data in windows:
        if len(w_data) == 0:
            w_rms.append(0.0)
            w_mean.append(0.0)
        else:
            w_rms.append(np.sqrt(np.mean(w_data**2)))
            w_mean.append(np.mean(w_data))

    feats[f"{sensor_name}_win_rms_mean"] = np.mean(w_rms)
    feats[f"{sensor_name}_win_rms_std"] = np.std(w_rms)
    feats[f"{sensor_name}_win_rms_max"] = np.max(w_rms)
    feats[f"{sensor_name}_win_mean_std"] = np.std(w_mean)

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
        # Cite solution_lesson_node_00049: Decompose Time-Series into Trend and Texture

        # 0. Decomposition
        window = params["sg_window_length"]
        poly = params["sg_polyorder"]
        # Trend (Low-frequency)
        trend = scipy.signal.savgol_filter(
            signal, window_length=window, polyorder=poly, deriv=0
        )
        # Texture (High-frequency residuals)
        texture = signal - trend

        # 1. Kinematic Features (on Trend/Signal via SavGol)
        # compute_kinematic_features uses savgol_filter internally on the input signal.
        # Passing 'signal' is correct because savgol fits the polynomial (trend) and takes derivative.
        k_feats = compute_kinematic_features(signal, sensor_col, params)
        features.update(k_feats)

        # 2. Texture Features (on Residuals)
        t_feats = compute_texture_features(texture, sensor_col, params)
        features.update(t_feats)

        # 3. Spectral/Intensity Features (on Raw Signal)
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
