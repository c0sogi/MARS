import os
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LinearRegression
from library.config import Config
from library.utils import load_sensor_data

# Ensure working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)


def impute_missing(df):
    """
    Fills missing values in the dataframe with the column-wise mean.
    """
    return df.fillna(df.mean())


def decompose_signal(sig):
    """
    Decomposes the signal into Trend, Texture (Residual), and Raw views.
    Trend is computed using a Savitzky-Golay filter.
    Texture is the residual (Raw - Trend).
    """
    # View A: Trend (Savitzky-Golay)
    # Order 2 is strictly used to avoid fitting noise acceleration
    trend = signal.savgol_filter(
        sig, window_length=Config.SG_WINDOW, polyorder=Config.SG_ORDER
    )

    # View C: Raw
    raw = sig

    # View B: Texture (Residuals)
    # Approximating texture as the high-frequency residual
    residual = raw - trend

    return trend, residual, raw


def compute_spectral_features(sig, fs):
    """
    Computes spectral band power using Welch's method.
    """
    features = {}
    freqs, psd = signal.welch(sig, fs=fs, nperseg=Config.NPERSEG)

    for band_idx, (low, high) in enumerate(Config.FREQ_BANDS):
        # Boolean mask for the frequency band
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # Sum PSD values in the band (approximation of integral)
        band_power = np.sum(psd[idx_band])
        features[f"band_{band_idx}_power"] = band_power

    return features


def compute_kinematics(trend):
    """
    Computes kinematic features (Velocity, Acceleration) and their moments from the Trend.
    """
    features = {}

    # Velocity (1st Derivative)
    vel = np.gradient(trend)
    # Acceleration (2nd Derivative)
    acc = np.gradient(vel)

    features["trend_mean"] = np.mean(trend)
    features["trend_std"] = np.std(trend)

    features["vel_std"] = np.std(vel)
    features["vel_skew"] = float(stats.skew(vel))

    features["acc_std"] = np.std(acc)
    features["acc_kurt"] = float(stats.kurtosis(acc))

    return features


def compute_texture_features(residual):
    """
    Computes complexity features from the Residual (Texture) component.
    """
    features = {}

    features["res_energy"] = np.sum(residual**2)
    # Simple entropy proxy using absolute values
    features["res_entropy"] = stats.entropy(np.abs(residual) + 1e-9)
    features["res_skew"] = float(stats.skew(residual))
    features["res_kurt"] = float(stats.kurtosis(residual))

    return features


def compute_trend_profiling(raw):
    """
    Computes trend-sensitive temporal profiling features.
    Splits signal into N windows, computes RMS/Mean per window,
    and calculates the linear slope of the RMS sequence.
    """
    features = {}

    window_size = len(raw) // Config.N_WINDOWS
    rms_sequence = []

    for w in range(Config.N_WINDOWS):
        start = w * window_size
        end = (w + 1) * window_size
        # Handle last window edge case
        if w == Config.N_WINDOWS - 1:
            chunk = raw[start:]
        else:
            chunk = raw[start:end]

        # Window Stats
        chunk_rms = np.sqrt(np.mean(chunk**2))
        chunk_mean = np.mean(chunk)

        # Feature Set A: Snapshots
        features[f"win_{w}_rms"] = chunk_rms
        features[f"win_{w}_mean"] = chunk_mean

        rms_sequence.append(chunk_rms)

    # Feature Set B: Dynamics (Slope/Volatility of RMS)
    if len(rms_sequence) > 1:
        X = np.arange(len(rms_sequence)).reshape(-1, 1)
        y = np.array(rms_sequence)
        reg = LinearRegression().fit(X, y)
        features["rms_slope"] = reg.coef_[0]
        features["rms_volatility"] = np.std(rms_sequence)
    else:
        features["rms_slope"] = 0.0
        features["rms_volatility"] = 0.0

    return features


def extract_segment_features(df, segment_id):
    """
    Aggregates features for all sensors in a segment using the Pyramidal Decomposition.
    """
    # Impute missing values
    df = impute_missing(df)

    features = {}
    features["segment_id"] = segment_id

    sensors = [c for c in df.columns if "sensor" in c]

    for sensor in sensors:
        sig = df[sensor].values.astype(np.float32)

        # 1. Decomposition
        trend, residual, raw = decompose_signal(sig)

        # 2. Level 1: Global Intensity & Spectrum (From Raw)
        features[f"{sensor}_min"] = np.min(raw)
        features[f"{sensor}_max"] = np.max(raw)
        features[f"{sensor}_ptp"] = np.ptp(raw)

        spec_feats = compute_spectral_features(raw, Config.FS)
        for k, v in spec_feats.items():
            features[f"{sensor}_{k}"] = v

        # 3. Level 2: Robust Kinematics (From Trend)
        kin_feats = compute_kinematics(trend)
        for k, v in kin_feats.items():
            features[f"{sensor}_{k}"] = v

        # 4. Level 3: Texture Complexity (From Residuals)
        tex_feats = compute_texture_features(residual)
        for k, v in tex_feats.items():
            features[f"{sensor}_{k}"] = v

        # 5. Level 4: Trend-Sensitive Temporal Profiling (From Raw)
        prof_feats = compute_trend_profiling(raw)
        for k, v in prof_feats.items():
            features[f"{sensor}_{k}"] = v

    return features


def process_data(mode="train", load_cached_data=True, debug_size=None):
    """
    Main data processing function.
    Loads metadata, iterates through files, extracts features, and caches results.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from parquet cache.
        debug_size (int, optional): Number of files to process for debugging.

    Returns:
        pd.DataFrame: The processed feature dataframe.
    """
    cache_file = os.path.join(Config.WORKING_DIR, f"{mode}_features.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        df = pd.read_parquet(cache_file)
        if debug_size:
            df = df.head(debug_size)
        return df

    # 2. Process from Scratch
    print(f"Processing {mode} data from scratch...")

    # Load Metadata
    meta_path = os.path.join(Config.METADATA_DIR, f"{mode}.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    if debug_size:
        meta_df = meta_df.head(debug_size)

    feature_list = []
    total = len(meta_df)

    for i, row in meta_df.iterrows():
        segment_id = row["segment_id"]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load sensor data
        sensor_df = load_sensor_data(file_path)

        if sensor_df.empty:
            print(f"Warning: Skipping empty or missing file {file_path}")
            continue

        try:
            # Extract Features
            feats = extract_segment_features(sensor_df, segment_id)

            # Add Target if available
            if "time_to_eruption" in row:
                feats["time_to_eruption"] = row["time_to_eruption"]

            feature_list.append(feats)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total} files.")

    # Create DataFrame
    result_df = pd.DataFrame(feature_list)

    # 3. Save Cache (only if not debugging)
    if not debug_size and not result_df.empty:
        print(f"Saving {mode} features to {cache_file}...")
        result_df.to_parquet(cache_file, index=False)

    return result_df
