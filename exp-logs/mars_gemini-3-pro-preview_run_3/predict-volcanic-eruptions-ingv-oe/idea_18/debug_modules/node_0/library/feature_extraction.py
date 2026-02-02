import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from library import config, signal_processing


def _get_trend_features(trend_signal):
    """
    Extracts kinematic features (velocity, acceleration) from the trend signal (View A).
    """
    features = {}

    # Compute derivatives
    velocity, acceleration = signal_processing.compute_derivatives(trend_signal)

    # Velocity Statistics
    features["vel_mean"] = np.mean(velocity)
    features["vel_std"] = np.std(velocity)
    features["vel_q01"] = np.quantile(velocity, 0.01)
    features["vel_q05"] = np.quantile(velocity, 0.05)
    features["vel_q95"] = np.quantile(velocity, 0.95)
    features["vel_q99"] = np.quantile(velocity, 0.99)

    # Acceleration Statistics
    features["acc_mean"] = np.mean(acceleration)
    features["acc_std"] = np.std(acceleration)
    features["acc_q01"] = np.quantile(acceleration, 0.01)
    features["acc_q05"] = np.quantile(acceleration, 0.05)
    features["acc_q95"] = np.quantile(acceleration, 0.95)
    features["acc_q99"] = np.quantile(acceleration, 0.99)

    return features


def _get_texture_features(texture_signal):
    """
    Extracts roughness and complexity features from the residual signal (View B).
    """
    features = {}

    # Higher-order moments
    features["skew"] = skew(texture_signal)
    features["kurtosis"] = kurtosis(texture_signal)

    # Wavelet Energy (Haar Level 1)
    wavelet_energy = signal_processing.compute_wavelet_energy(texture_signal)
    features["wavelet_energy_detail"] = wavelet_energy["energy_detail"]
    features["wavelet_energy_approx"] = wavelet_energy["energy_approx"]

    return features


def _get_spectral_features(raw_signal):
    """
    Extracts spectral features (Band Power, Centroid) from the raw signal (View C).
    """
    features = {}

    # Compute PSD
    freqs, psd = signal_processing.compute_welch_psd(raw_signal)

    # Band Powers
    band_powers = signal_processing.compute_band_powers(freqs, psd)
    features.update(band_powers)

    # Spectral Centroid
    # Sum(f * P(f)) / Sum(P(f))
    psd_sum = np.sum(psd)
    if psd_sum > 0:
        features["spectral_centroid"] = np.sum(freqs * psd) / psd_sum
    else:
        features["spectral_centroid"] = 0.0

    return features


def _get_temporal_distribution_features(raw_signal):
    """
    Extracts shift-invariant temporal profile features by aggregating window statistics (View C).
    """
    features = {}

    # Split signal into non-overlapping windows
    windows = np.array_split(raw_signal, config.N_TEMPORAL_WINDOWS)

    window_means = []
    window_rms = []

    for w in windows:
        stats = signal_processing.get_signal_stats(w)
        window_means.append(stats["mean"])
        window_rms.append(stats["rms"])

    # Convert to numpy arrays for aggregation
    window_means = np.array(window_means)
    window_rms = np.array(window_rms)

    # Aggregated Distributional Statistics (Shift-Invariant)
    # Volatility of the mean over time
    features["temp_mean_mean"] = np.mean(window_means)
    features["temp_mean_std"] = np.std(window_means)
    features["temp_mean_range"] = np.ptp(window_means)  # Peak-to-peak (max - min)

    # Volatility of the energy (RMS) over time
    features["temp_rms_mean"] = np.mean(window_rms)
    features["temp_rms_std"] = np.std(window_rms)
    features["temp_rms_range"] = np.ptp(window_rms)

    return features


def extract_segment_features(segment_df):
    """
    Processes a single segment DataFrame (10 sensors) and returns a flat feature dictionary.

    Args:
        segment_df (pd.DataFrame): DataFrame containing sensor readings.

    Returns:
        dict: Flattened dictionary of features.
    """
    row_features = {}

    for sensor in config.SENSORS:
        if sensor not in segment_df.columns:
            continue

        # 1. Preprocessing: Imputation
        # Using float32 to match loading type, but computations often upcast to float64 automatically
        raw_signal = segment_df[sensor].values
        # Fill missing values to preserve DC offset
        raw_signal = signal_processing.fill_missing_values(raw_signal)

        # 2. Orthogonal Decomposition
        # View A: Trend (Low Frequency / Kinematic)
        trend_signal = signal_processing.apply_savitzky_golay(raw_signal)

        # View B: Texture (High Frequency / Residual)
        texture_signal = raw_signal - trend_signal

        # View C: Energy (Raw Signal)
        # (Already loaded as raw_signal)

        # 3. Feature Extraction

        # Trend Features
        trend_feats = _get_trend_features(trend_signal)
        for k, v in trend_feats.items():
            row_features[f"{sensor}_trend_{k}"] = v

        # Texture Features
        texture_feats = _get_texture_features(texture_signal)
        for k, v in texture_feats.items():
            row_features[f"{sensor}_texture_{k}"] = v

        # Energy/Raw Intensity Features
        row_features[f"{sensor}_raw_min"] = np.min(raw_signal)
        row_features[f"{sensor}_raw_max"] = np.max(raw_signal)
        row_features[f"{sensor}_raw_ptp"] = np.ptp(raw_signal)

        # Spectral Features
        spectral_feats = _get_spectral_features(raw_signal)
        for k, v in spectral_feats.items():
            row_features[f"{sensor}_spec_{k}"] = v

        # Temporal Distribution Features
        temp_feats = _get_temporal_distribution_features(raw_signal)
        for k, v in temp_feats.items():
            row_features[f"{sensor}_temp_{k}"] = v

    return row_features


def process_dataset(metadata_df, dataset_name, load_cached_data=True, debug=False):
    """
    Orchestrates the feature extraction process for a dataset defined by metadata.
    Handles caching to Parquet files.

    Args:
        metadata_df (pd.DataFrame): Metadata containing 'segment_id' and 'file_path'.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, processes only a small subset of data.

    Returns:
        pd.DataFrame: DataFrame containing features and (if available) targets.
    """
    # Determine cache path
    cache_filename = f"{dataset_name}_features"
    if debug:
        cache_filename += "_debug"
    cache_path = os.path.join(config.WORKING_DIR, f"{cache_filename}.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    # 2. Process Data
    print(f"Processing {dataset_name} dataset (Debug={debug})...")

    # Subset for debug
    if debug:
        metadata_df = metadata_df.iloc[: config.DEBUG_SAMPLE_SIZE].copy()

    features_list = []

    # Iterate through metadata
    # Avoiding tqdm as per instructions to reduce log verbosity
    count = 0
    total = len(metadata_df)

    for idx, row in metadata_df.iterrows():
        segment_id = row["segment_id"]
        file_path = os.path.join(config.INPUT_DIR, row["file_path"])

        try:
            # Load sensor data
            # Use float32 to handle NaNs and optimize memory
            df = pd.read_csv(file_path, dtype="float32")

            # Extract features
            feats = extract_segment_features(df)
            feats["segment_id"] = segment_id

            # Add target if available
            if "time_to_eruption" in row:
                feats["time_to_eruption"] = row["time_to_eruption"]

            features_list.append(feats)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

        count += 1
        if count % 500 == 0:
            print(f"Processed {count}/{total} files...")

    # Create DataFrame
    features_df = pd.DataFrame(features_list)

    # 3. Save Cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Saving features to {cache_path}...")
    features_df.to_parquet(cache_path, index=False)

    return features_df
