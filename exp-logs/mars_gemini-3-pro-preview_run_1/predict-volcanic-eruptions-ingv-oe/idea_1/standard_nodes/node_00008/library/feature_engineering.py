import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import library.config as config


def extract_segment_features(file_path):
    """
    Reads a sensor segment CSV and calculates statistical features.

    Args:
        file_path (str): Path to the .csv file containing sensor data.

    Returns:
        dict: A dictionary containing flattened features for the segment.
              Returns None if the file cannot be read.
    """
    try:
        # Load data with float32 to optimize memory and handle potential NaNs
        df = pd.read_csv(file_path, dtype="float32")
    except FileNotFoundError:
        return None

    # Impute missing values with the mean of each column
    df = df.fillna(df.mean())

    features = {}

    # Iterate through each sensor defined in the config
    for sensor in config.SENSOR_COLS:
        if sensor in df.columns:
            series = df[sensor]

            # If a column is entirely NaN (rare), fill with 0 to avoid errors
            if series.isnull().all():
                series = series.fillna(0)

            # Convert to numpy for potentially faster operations on some stats
            arr = series.values

            # --- Central Tendency & Dispersion ---
            mean_val = np.mean(arr)
            features[f"{sensor}_mean"] = mean_val
            features[f"{sensor}_std"] = np.std(arr, ddof=1)  # Sample standard deviation

            # --- Extremes ---
            features[f"{sensor}_min"] = np.min(arr)
            features[f"{sensor}_max"] = np.max(arr)

            # --- Shape ---
            # Pandas skew/kurt are consistent with sample statistics
            features[f"{sensor}_skew"] = series.skew()
            features[f"{sensor}_kurt"] = series.kurtosis()

            # --- Quantiles ---
            # Computing multiple quantiles at once is efficient in pandas
            quantiles = series.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            features[f"{sensor}_q01"] = quantiles[0.01]
            features[f"{sensor}_q05"] = quantiles[0.05]
            features[f"{sensor}_q25"] = quantiles[0.25]
            features[f"{sensor}_q50"] = quantiles[0.50]
            features[f"{sensor}_q75"] = quantiles[0.75]
            features[f"{sensor}_q95"] = quantiles[0.95]
            features[f"{sensor}_q99"] = quantiles[0.99]

            # --- Dynamics ---
            # Mean Absolute Deviation: mean(|x - mean|)
            mad = np.mean(np.abs(arr - mean_val))
            features[f"{sensor}_mad"] = mad

            # --- Frequency Domain (FFT) ---
            # Compute Real FFT (magnitude)
            fft_vals = np.fft.rfft(arr)
            fft_mag = np.abs(fft_vals)

            # Skip DC component (index 0) for stats
            if len(fft_mag) > 1:
                fft_mag_no_dc = fft_mag[1:]
                features[f"{sensor}_fft_mean"] = np.mean(fft_mag_no_dc)
                features[f"{sensor}_fft_std"] = np.std(fft_mag_no_dc)
                features[f"{sensor}_fft_max"] = np.max(fft_mag_no_dc)
                features[f"{sensor}_fft_dom_freq_idx"] = np.argmax(fft_mag_no_dc)
            else:
                features[f"{sensor}_fft_mean"] = 0.0
                features[f"{sensor}_fft_std"] = 0.0
                features[f"{sensor}_fft_max"] = 0.0
                features[f"{sensor}_fft_dom_freq_idx"] = 0.0

            # --- Zero Crossing Rate ---
            # Use raw data for ZCR to capture amplitude-gated frequency (Cite solution_lesson_node_00007)
            zcr = ((arr[:-1] * arr[1:]) < 0).sum()
            features[f"{sensor}_zcr"] = zcr

            # --- Structural ---
            # Total Variation (Cite solution_lesson_node_00006)
            total_variation = np.sum(np.abs(np.diff(arr)))
            features[f"{sensor}_total_variation"] = total_variation

        else:
            # If a sensor column is missing from the CSV, fill features with NaN
            for stat in config.STATS_COLS:
                features[f"{sensor}_{stat}"] = np.nan

    return features


def _process_row(row, input_dir):
    """
    Helper function to process a single row from the metadata DataFrame.
    Used for parallel execution.
    """
    segment_id = row["segment_id"]
    rel_path = row["file_path"]
    full_path = os.path.join(input_dir, rel_path)

    # Extract features
    feats = extract_segment_features(full_path)

    if feats is not None:
        # Add metadata to the feature dictionary
        feats["segment_id"] = int(segment_id)
        if "time_to_eruption" in row:
            feats["time_to_eruption"] = row["time_to_eruption"]
        return feats
    return None


def build_feature_matrix(metadata_df, dataset_name, load_cached_data=True):
    """
    Orchestrates the feature extraction process for a dataset (train/val/test).
    Handles caching to avoid re-computation.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'segment_id' and 'file_path'.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for cache naming.
        load_cached_data (bool): If True, attempts to load from disk before computing.

    Returns:
        pd.DataFrame: A DataFrame containing segment_ids, targets (if available), and extracted features.
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    cache_file = os.path.join(config.CACHE_DIR, f"{dataset_name}_features.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached features for {dataset_name} from {cache_file}")
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Extracting features for {dataset_name} ({len(metadata_df)} segments)...")

    # Convert metadata to list of dictionaries for iteration
    rows = metadata_df.to_dict("records")

    # Use joblib for parallel processing
    # n_jobs is set in config (12 vCPUs)
    results = Parallel(n_jobs=config.N_JOBS)(
        delayed(_process_row)(row, config.INPUT_DIR) for row in rows
    )

    # Filter out any failed extractions (None)
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        raise ValueError(f"No features could be extracted for {dataset_name}.")

    feature_df = pd.DataFrame(valid_results)

    # 3. Save to cache
    print(f"Saving features to {cache_file}")
    feature_df.to_parquet(cache_file, index=False)

    return feature_df
