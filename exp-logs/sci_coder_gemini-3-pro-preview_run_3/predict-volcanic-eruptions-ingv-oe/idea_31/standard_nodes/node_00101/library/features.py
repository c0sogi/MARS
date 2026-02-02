import os
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from joblib import Parallel, delayed
import library.config as config


def apply_savitzky_golay(signal, window, order):
    """
    Applies a Savitzky-Golay filter to smooth the signal (View A: Trend).
    Strictly uses Order 2 to avoid overfitting acceleration noise.
    """
    return scipy.signal.savgol_filter(signal, window_length=window, polyorder=order)


def apply_dwt(signal, trend):
    """
    Extracts the Texture View (View B).
    Computes the Residuals (Raw - Trend) which represent the high-frequency
    detail coefficients of the signal.
    """
    return signal - trend


def compute_spectral_features(signal, fs, nperseg, bands):
    """
    Computes High-Resolution Spectral features using Welch's Method.
    Integrates Power Spectral Density (PSD) over specific frequency bands.
    """
    features = {}
    f, Pxx = scipy.signal.welch(signal, fs=fs, nperseg=nperseg)

    for band, (low, high) in bands.items():
        mask = (f >= low) & (f <= high)
        if np.any(mask):
            # Integrate area under the curve for the band
            features[f"spec_{band}"] = np.trapz(Pxx[mask], f[mask])
        else:
            features[f"spec_{band}"] = 0.0
    return features


def compute_differential_profiling(signal, num_windows):
    """
    Implements Hierarchical Aggregated Temporal Profiling.
    Splits signal into windows, computes snapshots (RMS, Mean),
    and calculates aggregated statistics across these windows to ensure shift invariance.
    Cite solution_lesson_node_00077, solution_lesson_node_00050
    """
    features = {}
    # Split signal into N non-overlapping windows
    windows = np.array_split(signal, num_windows)
    rms_vals = []
    mean_vals = []

    # Step 1: Snapshots
    for i, w in enumerate(windows):
        if len(w) > 0:
            rms = np.sqrt(np.mean(w**2))
            mean_val = np.mean(w)
        else:
            rms = 0.0
            mean_val = 0.0

        rms_vals.append(rms)
        mean_vals.append(mean_val)

    # Step 2: Aggregated Statistics (Shift Invariant)
    # Instead of specific window indices, we describe the distribution of the windows
    rms_vals = np.array(rms_vals)
    mean_vals = np.array(mean_vals)

    features["win_rms_mean"] = np.mean(rms_vals)
    features["win_rms_std"] = np.std(rms_vals)
    features["win_rms_min"] = np.min(rms_vals)
    features["win_rms_max"] = np.max(rms_vals)
    features["win_rms_range"] = np.ptp(rms_vals)

    features["win_mean_mean"] = np.mean(mean_vals)
    features["win_mean_std"] = np.std(mean_vals)
    features["win_mean_min"] = np.min(mean_vals)
    features["win_mean_max"] = np.max(mean_vals)
    features["win_mean_range"] = np.ptp(mean_vals)

    return features


def extract_segment_features(df):
    """
    Orchestrates the feature extraction pipeline for a single data segment.
    Iterates over sensors and applies the Pyramidal Decomposition.
    """
    # Impute missing values with column means (Segment-wise)
    df = df.fillna(df.mean())

    features = {}
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    for sensor in sensor_cols:
        x = df[sensor].values.astype(np.float32)

        # --- View A: Trend (Savitzky-Golay) ---
        trend = apply_savitzky_golay(x, config.SG_WINDOW, config.SG_ORDER)

        # --- View B: Texture (Residuals) ---
        residual = apply_dwt(x, trend)

        # --- Feature Group 1: Kinematics (from View A) ---
        vel = np.diff(trend)
        acc = np.diff(vel)

        # Quantiles for granular distribution shape (Cite solution_lesson_node_00059, solution_lesson_node_00026)
        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        for name, sig in [("trend", trend), ("vel", vel), ("acc", acc)]:
            features[f"{sensor}_{name}_mean"] = np.mean(sig)
            features[f"{sensor}_{name}_std"] = np.std(sig)
            features[f"{sensor}_{name}_skew"] = float(scipy.stats.skew(sig))
            features[f"{sensor}_{name}_kurt"] = float(scipy.stats.kurtosis(sig))

            # Add granular quantiles
            q_vals = np.quantile(sig, quantiles)
            for q, val in zip(quantiles, q_vals):
                features[f"{sensor}_{name}_q{int(q*100):02d}"] = val

        # --- Feature Group 2: Texture Stats (from View B) ---
        features[f"{sensor}_res_energy"] = np.sum(residual**2)
        features[f"{sensor}_res_skew"] = float(scipy.stats.skew(residual))
        features[f"{sensor}_res_kurt"] = float(scipy.stats.kurtosis(residual))

        # Removed Entropy (Cite solution_lesson_node_00012)

        # --- Feature Group 3: Intensity (from View C - Raw) ---
        features[f"{sensor}_min"] = np.min(x)
        features[f"{sensor}_max"] = np.max(x)
        features[f"{sensor}_ptp"] = np.ptp(x)  # Peak-to-Peak

        # --- Feature Group 4: High-Res Spectral (from View C) ---
        spec_feats = compute_spectral_features(
            x, config.FS, config.WELCH_NPERSEG, config.FREQ_BANDS
        )
        for k, v in spec_feats.items():
            features[f"{sensor}_{k}"] = v

        # --- Feature Group 5: Differential Temporal Profiling (from View C) ---
        diff_feats = compute_differential_profiling(x, config.NUM_TEMPORAL_WINDOWS)
        for k, v in diff_feats.items():
            features[f"{sensor}_{k}"] = v

    return features


def _process_file_wrapper(row):
    """
    Helper function to process a single file row from metadata.
    Used for parallel execution.
    """
    try:
        file_path = os.path.join(config.INPUT_DIR, row["file_path"])
        # Load with float32 to handle potential nulls and optimize memory usage
        df = pd.read_csv(file_path, dtype="float32")

        feats = extract_segment_features(df)

        # Add metadata
        feats["segment_id"] = int(row["segment_id"])
        if "time_to_eruption" in row:
            feats["time_to_eruption"] = row["time_to_eruption"]

        return feats
    except Exception as e:
        print(f"Error processing segment {row.get('segment_id', 'unknown')}: {e}")
        return None


def process_data(metadata_path, cache_name, load_cached_data=True):
    """
    Loads metadata, processes sensor files in parallel to extract features,
    and manages caching of the resulting dataframe.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_name (str): Filename for the parquet cache.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed features dataframe.
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(config.WORKING_DIR, cache_name)

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Compute if cache missing or forced reload
    print(f"Processing data from {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    # Parallel execution using joblib
    # Using n_jobs=12 as per available vCPUs
    results = Parallel(n_jobs=12, verbose=0)(
        delayed(_process_file_wrapper)(row) for _, row in meta_df.iterrows()
    )

    # Filter out failures
    results = [r for r in results if r is not None]

    if not results:
        raise ValueError("No data was successfully processed.")

    df_features = pd.DataFrame(results)

    # 3. Save to cache
    print(f"Saving features to {cache_path}")
    df_features.to_parquet(cache_path, index=False)

    return df_features
