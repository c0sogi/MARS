import os
import numpy as np
import pandas as pd
from scipy import stats, signal
import library.config as config


def preprocess_signal(df):
    """
    Preprocesses the sensor dataframe by filling missing values and removing linear trends.

    Args:
        df (pd.DataFrame): Raw sensor data.

    Returns:
        pd.DataFrame: Detrended and imputed data.
    """
    # Impute missing values with the mean of each column
    # If a column is all NaN, mean() is NaN, so we fill those with 0
    df_filled = df.fillna(df.mean()).fillna(0)

    # Detrend the signal (remove linear trend) to avoid spectral leakage
    # signal.detrend returns a numpy array, so we reconstruct the DataFrame
    detrended_data = signal.detrend(df_filled, axis=0)
    df_detrended = pd.DataFrame(detrended_data, columns=df.columns)

    return df_detrended


def compute_time_stats(x):
    """
    Computes global statistical features in the time domain.
    """
    mean_val = np.mean(x)
    std_val = np.std(x)

    # Dispersion and Shape
    mad_val = np.mean(np.abs(x - mean_val))
    skew_val = stats.skew(x)
    kurt_val = stats.kurtosis(x)

    # Quantiles
    q01 = np.quantile(x, 0.01)
    q05 = np.quantile(x, 0.05)
    q95 = np.quantile(x, 0.95)
    q99 = np.quantile(x, 0.99)
    iqr_val = np.subtract(*np.percentile(x, [75, 25]))
    rms_val = np.sqrt(np.mean(x**2))

    return {
        "mean": mean_val,
        "median": np.median(x),
        "std": std_val,
        "min": np.min(x),
        "max": np.max(x),
        "mad": mad_val,
        "skew": skew_val,
        "kurtosis": kurt_val,
        "q01": q01,
        "q05": q05,
        "q95": q95,
        "q99": q99,
        "iqr": iqr_val,
        "rms": rms_val,
    }


def compute_grad_stats(x):
    """
    Computes statistics on the first derivative (gradient) of the signal.
    Cite Lesson 2: Dimensionality Reduction via Statistical Moments.
    """
    dx = np.diff(x)
    return {
        "grad_mean": np.mean(dx),
        "grad_std": np.std(dx),
        "grad_skew": stats.skew(dx),
        "grad_kurt": stats.kurtosis(dx),
    }


def compute_freq_stats(x, fs=100):
    """
    Computes spectral features using FFT.
    Assumes a sampling rate (fs) of 100 Hz.
    """
    # Compute Power Spectral Density (PSD)
    f, Pxx = signal.periodogram(x, fs)

    # Handle silence/flatline
    Pxx_sum = np.sum(Pxx)
    if Pxx_sum == 0:
        return {
            "spec_centroid": 0,
            "peak_freq": 0,
            "spec_entropy": 0,
            "power_low": 0,
            "power_mid": 0,
            "power_high": 0,
        }

    # Normalize for distribution features
    Pxx_norm = Pxx / Pxx_sum

    # Spectral Centroid (Center of Mass of the spectrum)
    centroid = np.sum(f * Pxx_norm)

    # Peak Frequency (Dominant harmonic)
    peak_freq = f[np.argmax(Pxx)]

    # Spectral Entropy (Measure of complexity/randomness)
    spec_entropy = stats.entropy(Pxx_norm)

    # Spectral Shape (Cite Lesson 5: Augmenting Global Aggregates)
    spec_skew = stats.skew(Pxx_norm)
    spec_kurt = stats.kurtosis(Pxx_norm)

    # Band Power Aggregation
    # Low: 0.1 - 5 Hz (Long period)
    # Mid: 5 - 15 Hz (Typical tremor)
    # High: 15 - 50 Hz (High freq events)
    mask_low = (f >= 0.1) & (f < 5.0)
    mask_mid = (f >= 5.0) & (f < 15.0)
    mask_high = f >= 15.0

    return {
        "spec_centroid": centroid,
        "peak_freq": peak_freq,
        "spec_entropy": spec_entropy,
        "spec_skew": spec_skew,
        "spec_kurt": spec_kurt,
        "power_low": np.sum(Pxx[mask_low]),
        "power_mid": np.sum(Pxx[mask_mid]),
        "power_high": np.sum(Pxx[mask_high]),
    }


def compute_windowed_stats(x, num_windows):
    """
    Splits the signal into non-overlapping windows and computes stats
    to capture temporal evolution/acceleration.
    """
    splits = np.array_split(x, num_windows)
    feats = {}
    for i, chunk in enumerate(splits):
        feats[f"win{i}_mean"] = np.mean(chunk)
        feats[f"win{i}_std"] = np.std(chunk)
    return feats


def extract_segment_features(df):
    """
    Extracts a flattened feature vector from a raw sensor dataframe.
    Combines Time, Frequency, and Windowed features for all sensors.
    """
    # Preprocessing
    df_clean = preprocess_signal(df)

    feature_dict = {}

    for sensor in config.SENSOR_COLS:
        if sensor not in df_clean.columns:
            continue

        x = df_clean[sensor].values

        # 1. Time Domain
        t_stats = compute_time_stats(x)
        for k, v in t_stats.items():
            feature_dict[f"{sensor}_{k}"] = v

        # 2. Frequency Domain
        f_stats = compute_freq_stats(x, fs=100)
        for k, v in f_stats.items():
            feature_dict[f"{sensor}_{k}"] = v

        # 3. Gradient Statistics (Cite Lesson 2)
        g_stats = compute_grad_stats(x)
        for k, v in g_stats.items():
            feature_dict[f"{sensor}_{k}"] = v

        # 4. Temporal Evolution (Windowing)
        w_stats = compute_windowed_stats(x, config.NUM_WINDOWS)
        for k, v in w_stats.items():
            feature_dict[f"{sensor}_{k}"] = v

    return pd.Series(feature_dict)


def process_dataset(metadata_path, load_cached_data=True, save_name="dataset_features"):
    """
    Orchestrates the feature engineering pipeline for a dataset defined by metadata.
    Implements Parquet-based caching.

    Args:
        metadata_path (str): Path to the metadata CSV (train/val/test).
        load_cached_data (bool): Whether to attempt loading from cache.
        save_name (str): Filename for the cached parquet file.

    Returns:
        pd.DataFrame: The processed features with segment_id and target (if available).
    """
    # Ensure cache directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(config.WORKING_DIR, f"{save_name}.parquet")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Compute from Scratch
    print(f"Processing features for {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file missing: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)
    features_list = []

    for idx, row in meta_df.iterrows():
        segment_id = row["segment_id"]
        file_path = os.path.join(config.INPUT_DIR, row["file_path"])

        try:
            # Load raw data (float32 for memory efficiency)
            df_raw = pd.read_csv(file_path, dtype="float32")

            # Extract features
            feats = extract_segment_features(df_raw)

            # Append Identity and Target
            feats["segment_id"] = segment_id
            if "time_to_eruption" in row:
                feats["time_to_eruption"] = row["time_to_eruption"]

            features_list.append(feats)

        except Exception as e:
            print(f"Error processing segment {segment_id} at {file_path}: {e}")

    # Compile DataFrame
    result_df = pd.DataFrame(features_list)

    # 3. Save to Cache
    result_df.to_parquet(cache_path, index=False)
    print(f"Saved processed features to {cache_path}")

    return result_df
