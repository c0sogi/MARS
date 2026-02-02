import os
import numpy as np
import pandas as pd
from scipy import signal, stats
from library.config import Config


def get_trend_view(data):
    """
    Applies Savitzky-Golay filter to isolate low-frequency baseline drift.
    """
    try:
        return signal.savgol_filter(
            data, window_length=Config.SG_WINDOW_SIZE, polyorder=Config.SG_POLY_ORDER
        )
    except Exception:
        # Fallback for very short signals or errors
        return data


def get_texture_view(raw, trend):
    """
    Computes residuals (Raw - Trend) to isolate texture/noise.
    """
    return raw - trend


def compute_dense_quantiles(data, prefix):
    """
    Computes a dense grid of quantiles for a given signal.
    """
    features = {}
    for q in Config.QUANTILES:
        key = f"{prefix}_q{int(q*100)}"
        features[key] = np.quantile(data, q)
    return features


def compute_texture_features(data, prefix):
    """
    Computes texture features (RMS, Entropy) on residuals.
    Acts as a proxy for Wavelet features to characterize signal roughness.
    """
    features = {}
    # RMS Energy of residuals
    features[f"{prefix}_rms"] = np.sqrt(np.mean(data**2))

    # Histogram Entropy
    # Adding small epsilon to avoid log(0)
    hist_counts, _ = np.histogram(data, bins=50, density=True)
    features[f"{prefix}_entropy"] = stats.entropy(hist_counts + 1e-10)

    return features


def compute_spectral_features(data, prefix):
    """
    Computes PSD Band Power using Welch's Method.
    """
    features = {}
    # fs=100Hz assumption for 10-minute segments with ~60k samples
    f, Pxx = signal.welch(data, fs=100, nperseg=256)

    # Band Power Integration
    features[f"{prefix}_band_low"] = np.sum(Pxx[(f >= 0) & (f < 5)])
    features[f"{prefix}_band_mid"] = np.sum(Pxx[(f >= 5) & (f < 15)])
    features[f"{prefix}_band_high"] = np.sum(Pxx[(f >= 15) & (f <= 50)])

    return features


def compute_temporal_features(data, prefix):
    """
    Computes features over non-overlapping temporal windows to capture evolution.
    """
    features = {}
    # Split into windows
    windows = np.array_split(data, Config.NUM_TEMPORAL_WINDOWS)
    for i, w in enumerate(windows):
        features[f"{prefix}_w{i}_rms"] = np.sqrt(np.mean(w**2))
    return features


def extract_features_for_segment(df):
    """
    Main feature extraction function for a single segment DataFrame.
    Implements the Hybrid-Transform Decomposition strategy.

    Args:
        df (pd.DataFrame): Raw sensor data for one segment.

    Returns:
        dict: Flattened dictionary of extracted features.
    """
    # 1. Imputation: Fill NaNs with column means to preserve DC offsets
    # Fallback to 0 if column is all-NaN (Cite debug_lesson_1)
    df = df.fillna(df.mean()).fillna(0)

    features = {}

    for sensor in Config.SENSOR_COLS:
        if sensor not in df.columns:
            continue

        # Ensure float32 for consistency
        raw = df[sensor].values.astype(np.float32)

        # --- View A: Trend (Savitzky-Golay) ---
        trend = get_trend_view(raw)

        # Kinematics (Derivatives of Trend)
        vel = np.gradient(trend)
        acc = np.gradient(vel)

        # Features from Trend View (Dense Quantiles)
        features.update(compute_dense_quantiles(trend, f"{sensor}_trend"))
        features.update(compute_dense_quantiles(vel, f"{sensor}_vel"))
        features.update(compute_dense_quantiles(acc, f"{sensor}_acc"))

        # --- View B: Texture (Residuals) ---
        resid = get_texture_view(raw, trend)
        features.update(compute_texture_features(resid, f"{sensor}_resid"))

        # --- View C: Raw / Energy ---
        # Absolute stats
        features[f"{sensor}_min"] = np.min(raw)
        features[f"{sensor}_max"] = np.max(raw)
        features[f"{sensor}_ptp"] = np.ptp(raw)

        # Spectral Structure
        features.update(compute_spectral_features(raw, sensor))

        # Temporal Evolution
        features.update(compute_temporal_features(raw, sensor))

    return features


def process_data(meta_path, load_cached_data=True, is_test=False, debug_size=None):
    """
    Loads metadata, processes all segments to extract features, and handles caching.

    Args:
        meta_path (str): Path to the metadata CSV (train.csv, val.csv, or test.csv).
        load_cached_data (bool): Whether to attempt loading from cache.
        is_test (bool): Whether processing test data (no target column).
        debug_size (int, optional): Limit number of files processed for debugging.

    Returns:
        pd.DataFrame: Processed features dataframe.
    """
    # Ensure working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Construct cache filename based on metadata filename and debug flag
    meta_name = os.path.basename(meta_path).replace(".csv", "")
    debug_suffix = f"_debug_{debug_size}" if debug_size else ""
    cache_filename = f"{meta_name}_features{debug_suffix}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    # 2. Process from Scratch
    print(f"Processing data from {meta_path} (Debug Size: {debug_size})...")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    if debug_size is not None:
        meta_df = meta_df.head(debug_size)

    feature_list = []

    # Iterate through metadata
    for idx, row in meta_df.iterrows():
        # Construct full file path
        # Metadata contains relative path (e.g., "train/123.csv")
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            if not os.path.exists(file_path):
                print(f"Warning: Data file not found {file_path}")
                continue

            # Load sensor data
            df = pd.read_csv(file_path, dtype="float32")

            # Extract features
            feats = extract_features_for_segment(df)

            # Add metadata identifiers
            feats["segment_id"] = int(row["segment_id"])

            if not is_test and "time_to_eruption" in row:
                feats["time_to_eruption"] = row["time_to_eruption"]

            feature_list.append(feats)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not feature_list:
        print("Warning: No features extracted.")
        return pd.DataFrame()

    # Create DataFrame
    features_df = pd.DataFrame(feature_list)

    # 3. Save to Cache
    print(f"Saving features to {cache_path}...")
    features_df.to_parquet(cache_path, index=False)

    return features_df
