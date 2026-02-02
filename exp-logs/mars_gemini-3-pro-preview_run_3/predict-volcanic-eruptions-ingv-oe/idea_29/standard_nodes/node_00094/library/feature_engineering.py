import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal

try:
    import pywt
except ImportError:
    pywt = None
from joblib import Parallel, delayed
from library.config import Config


def calculate_entropy(x):
    """Calculate Shannon Entropy of the energy distribution of a signal."""
    energy = np.sum(x**2)
    if energy == 0:
        return 0.0
    # Probability distribution of energy
    p = (x**2) / energy
    # Filter zeros to avoid log(0)
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def extract_stream_a_features(sensor_id, raw_signal, trend_signal):
    """
    Stream A: Trend Extraction & Kinematics.
    Computes derivatives and granular quantiles on the smoothed trend.
    Replaces unstable moments with robust quantiles (Cite solution_lesson_node_00026).
    """
    features = {}
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

    # 0th Derivative (Position/Trend)
    features[f"{sensor_id}_trend_mean"] = np.mean(trend_signal)
    features[f"{sensor_id}_trend_std"] = np.std(trend_signal)
    for q in quantiles:
        features[f"{sensor_id}_trend_q{int(q*100)}"] = np.quantile(trend_signal, q)

    # 1st Derivative (Velocity)
    velocity = np.gradient(trend_signal)
    features[f"{sensor_id}_vel_mean"] = np.mean(velocity)
    features[f"{sensor_id}_vel_std"] = np.std(velocity)
    for q in quantiles:
        features[f"{sensor_id}_vel_q{int(q*100)}"] = np.quantile(velocity, q)

    # 2nd Derivative (Acceleration)
    acceleration = np.gradient(velocity)
    features[f"{sensor_id}_acc_mean"] = np.mean(acceleration)
    features[f"{sensor_id}_acc_std"] = np.std(acceleration)
    for q in quantiles:
        features[f"{sensor_id}_acc_q{int(q*100)}"] = np.quantile(acceleration, q)

    return features


def extract_stream_b_features(sensor_id, raw_signal, trend_signal):
    """
    Stream B: Texture Extraction via Wavelets.
    Applies DWT to residuals (Raw - Trend) and extracts stats from detail coefficients.
    """
    features = {}

    if pywt is None:
        return features

    # Compute Residuals
    residuals = raw_signal - trend_signal

    # Discrete Wavelet Transform
    # We use the first level detail coefficients for high-freq texture
    try:
        coeffs = pywt.wavedec(residuals, Config.DWT_WAVELET, level=1)
        # coeffs[0] is approx (cA), coeffs[1] is detail (cD). We want detail.
        detail_coeffs = coeffs[-1]
    except Exception:
        # Fallback if signal is too short or other error
        detail_coeffs = residuals

    features[f"{sensor_id}_txt_energy"] = np.sum(detail_coeffs**2)
    features[f"{sensor_id}_txt_entropy"] = calculate_entropy(detail_coeffs)
    features[f"{sensor_id}_txt_skew"] = stats.skew(detail_coeffs)
    features[f"{sensor_id}_txt_kurt"] = stats.kurtosis(detail_coeffs)

    return features


def extract_stream_c_features(sensor_id, raw_signal):
    """
    Stream C: Raw Signal Analysis.
    High-Res Spectral Structure, Temporal Profiling, and Global Extremes.
    """
    features = {}

    # 1. Global Extremes
    features[f"{sensor_id}_min"] = np.min(raw_signal)
    features[f"{sensor_id}_max"] = np.max(raw_signal)
    features[f"{sensor_id}_p2p"] = np.ptp(raw_signal)

    # 2. High-Resolution Spectral Structure (Welch PSD)
    f, Pxx = signal.welch(
        raw_signal,
        fs=Config.FS,
        nperseg=Config.PSD_NPERSEG,
        noverlap=Config.PSD_NOVERLAP,
    )

    # Integrate Band Power
    for band_name, (low, high) in Config.PSD_BANDS.items():
        idx = (f >= low) & (f <= high)
        if np.sum(idx) > 0:
            power = np.trapz(Pxx[idx], f[idx])
        else:
            power = 0.0
        features[f"{sensor_id}_psd_{band_name}"] = power

    # 3. Shift-Invariant Temporal Profiling (Cite solution_lesson_node_00050)
    # Split signal into non-overlapping windows and aggregate stats across them
    windows = np.array_split(raw_signal, Config.TEMPORAL_WINDOWS)

    win_means = [np.mean(w) for w in windows]
    win_rmss = [np.sqrt(np.mean(w**2)) for w in windows]

    # Aggregate window statistics
    features[f"{sensor_id}_win_mean_mean"] = np.mean(win_means)
    features[f"{sensor_id}_win_mean_std"] = np.std(win_means)
    features[f"{sensor_id}_win_rms_mean"] = np.mean(win_rmss)
    features[f"{sensor_id}_win_rms_std"] = np.std(win_rmss)

    return features


def process_segment(segment_id, file_path):
    """
    Master function to process a single sensor data file.
    Implements the Tri-Stream decomposition for all 10 sensors.
    """
    try:
        # Load Data
        full_path = os.path.join(Config.INPUT_DIR, file_path)
        # Load as float32 to handle NaNs and memory
        df = pd.read_csv(full_path, dtype="float32")

        # Imputation: Fill NaNs with column mean to preserve DC offset
        df = df.fillna(df.mean())

        segment_features = {"segment_id": int(segment_id)}

        for sensor in Config.SENSOR_COLS:
            if sensor not in df.columns:
                continue

            raw_signal = df[sensor].values

            # --- Stream A: Trend ---
            # Savitzky-Golay Filter
            try:
                trend_signal = signal.savgol_filter(
                    raw_signal,
                    window_length=Config.SG_WINDOW,
                    polyorder=Config.SG_POLYORDER,
                )
            except ValueError:
                # Fallback for very short signals (unlikely given dataset)
                trend_signal = raw_signal

            feat_a = extract_stream_a_features(sensor, raw_signal, trend_signal)
            segment_features.update(feat_a)

            # --- Stream B: Texture ---
            feat_b = extract_stream_b_features(sensor, raw_signal, trend_signal)
            segment_features.update(feat_b)

            # --- Stream C: Raw/Spectral ---
            feat_c = extract_stream_c_features(sensor, raw_signal)
            segment_features.update(feat_c)

        return segment_features

    except Exception as e:
        print(f"Error processing segment {segment_id} ({file_path}): {e}")
        return None


def generate_features(metadata_path, output_filename, load_cached_data=True):
    """
    Generates feature dataset from metadata.
    Handles caching, parallel processing, and DataFrame construction.

    Args:
        metadata_path (str): Path to the metadata CSV (train/val/test).
        output_filename (str): Name of the output parquet file (e.g., 'train_features.parquet').
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: The processed feature dataframe.
    """
    cache_path = os.path.join(Config.CACHE_DIR, output_filename)

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Generating features for {output_filename}...")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    # 3. Parallel Processing
    # Use joblib to process files in parallel
    results = Parallel(n_jobs=Config.LGBM_PARAMS["n_jobs"], verbose=0)(
        delayed(process_segment)(row["segment_id"], row["file_path"])
        for _, row in meta_df.iterrows()
    )

    # Filter out None results (errors)
    results = [r for r in results if r is not None]

    # 4. Construct DataFrame
    feature_df = pd.DataFrame(results)

    # 5. Merge Targets if available
    if "time_to_eruption" in meta_df.columns:
        # Merge on segment_id to ensure alignment
        feature_df = feature_df.merge(
            meta_df[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
        )

    # 6. Save to Cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    feature_df.to_parquet(cache_path, index=False)
    print(f"Features saved to {cache_path}")

    return feature_df
