import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from library.config import Config


def load_sensor_data(file_path: str) -> pd.DataFrame:
    """
    Reads a sensor data CSV file and fills NaN values with zeros.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The sensor data with NaNs filled.
    """
    try:
        df = pd.read_csv(file_path)
        df = df.fillna(0)
        return df
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        # Return an empty dataframe with correct columns if read fails,
        # though in this dataset structure we expect valid files.
        return pd.DataFrame()


def extract_statistical_features(df_raw: pd.DataFrame) -> dict:
    """
    Computes statistical features for each sensor.

    Args:
        df_raw (pd.DataFrame): Raw sensor data (may contain NaNs).

    Returns:
        dict: Dictionary containing flattened features for all sensors.
    """
    features = {}

    # 1. NaN Counts (computed on raw data before filling)
    nan_counts = df_raw.isna().sum()

    # 2. Fill NaNs for statistical computation
    df_filled = df_raw.fillna(0)

    # Iterate over sensor columns
    sensor_cols = [c for c in df_filled.columns if "sensor_" in c]

    for col in sensor_cols:
        series = df_filled[col]

        # Central Tendency and Dispersion
        features[f"{col}_mean"] = series.mean()
        features[f"{col}_std"] = series.std()
        features[f"{col}_min"] = series.min()
        features[f"{col}_max"] = series.max()

        # Shape Statistics
        features[f"{col}_skew"] = series.skew()
        features[f"{col}_kurtosis"] = series.kurtosis()

        # Quantiles (Distribution shape)
        # 1%, 5%, 25%, 50%, 75%, 95%, 99%
        quantiles = series.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        features[f"{col}_q01"] = quantiles[0.01]
        features[f"{col}_q05"] = quantiles[0.05]
        features[f"{col}_q25"] = quantiles[0.25]
        features[f"{col}_q50"] = quantiles[0.50]
        features[f"{col}_q75"] = quantiles[0.75]
        features[f"{col}_q95"] = quantiles[0.95]
        features[f"{col}_q99"] = quantiles[0.99]

        # Data Quality
        features[f"{col}_nan_count"] = nan_counts[col]

    return features


def generate_log_mel_spectrogram(data) -> torch.Tensor:
    """
    Generates a Log-Mel Spectrogram from sensor data using torchaudio.

    Args:
        data (pd.DataFrame or np.ndarray): Input data of shape (Time, Channels).
                                           Expected to be filled/cleaned.

    Returns:
        torch.Tensor: Log-Mel Spectrogram of shape (Channels, n_mels, time_steps).
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Convert to tensor and transpose to (Channels, Time)
    # Data is usually int16 range but normalized float, convert to float32
    waveform = torch.tensor(data, dtype=torch.float32).transpose(0, 1)

    # Define Transforms
    # We instantiate transforms here to ensure they use the current Config
    # and to keep the function stateless.
    mel_transform = T.MelSpectrogram(
        sample_rate=Config.SAMPLING_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        normalized=False,
    )

    db_transform = T.AmplitudeToDB(top_db=Config.TOP_DB)

    # Apply Transforms
    # waveform shape: [Channels, Time] -> spec shape: [Channels, n_mels, Time_frames]
    mel_spec = mel_transform(waveform)
    log_mel_spec = db_transform(mel_spec)

    return log_mel_spec


def process_and_cache_features(
    metadata_path: str, output_path: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Orchestrates the feature extraction process:
    1. Checks for cached Parquet file.
    2. If not found or forced reload, iterates through metadata.
    3. Loads raw data, computes statistics, and aggregates results.
    4. Saves results to Parquet for future use.

    Args:
        metadata_path (str): Path to the metadata CSV (train/val/test).
        output_path (str): Destination path for the Parquet file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed feature set.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached features from {output_path}")
        return pd.read_parquet(output_path)

    print(f"Processing features from {metadata_path}...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # Check if target column exists (it won't for test set)
    has_target = "time_to_eruption" in df_meta.columns

    feature_rows = []

    # 3. Iterate and Process
    for _, row in df_meta.iterrows():
        segment_id = row["segment_id"]
        # Construct full path. Metadata paths are relative to input dir.
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue

        # Read raw to get NaNs, then compute stats
        # We read directly here to avoid double-reading if we used load_sensor_data
        try:
            df_raw = pd.read_csv(file_path)

            # Extract features
            feats = extract_statistical_features(df_raw)

            # Append ID and Target
            feats["segment_id"] = segment_id
            if has_target:
                feats["time_to_eruption"] = row["time_to_eruption"]

            feature_rows.append(feats)

        except Exception as e:
            print(f"Error processing segment {segment_id}: {e}")
            continue

    # 4. Save to Cache
    if not feature_rows:
        print("Warning: No features were extracted.")
        return pd.DataFrame()

    df_features = pd.DataFrame(feature_rows)
    df_features.to_parquet(output_path, index=False)

    print(f"Saved processed features to {output_path} ({len(df_features)} rows)")

    return df_features
