import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.stats import skew, kurtosis
from library.config import Config


def compute_statistics(df: pd.DataFrame) -> np.ndarray:
    """
    Computes statistical features for a given sensor segment.

    Args:
        df (pd.DataFrame): Dataframe containing sensor readings (shape: [60001, 10]).

    Returns:
        np.ndarray: A flat float32 array of features.
                    Shape: [NUM_SENSORS * NUM_STATS_PER_SENSOR] (100 features).
    """
    stats = []

    # Ensure we iterate in a fixed order: sensor_1 to sensor_10
    sensor_cols = [f"sensor_{i}" for i in range(1, Config.NUM_SENSORS + 1)]

    for col in sensor_cols:
        if col in df.columns:
            series = df[col]

            # 1. NaN Count (Reliability metric)
            n_nans = series.isna().sum()

            # 2. Fill NaNs with 0 for statistical calculation
            # (Consistent with spectrogram preprocessing)
            data = series.fillna(0).values

            # 3. Basic Stats
            mean = np.mean(data)
            std = np.std(data)

            # 4. Shape Stats
            # Handle constant arrays to avoid division by zero in skew/kurtosis
            if std < 1e-9:
                s_skew = 0.0
                s_kurt = 0.0
            else:
                s_skew = skew(data)
                s_kurt = kurtosis(data)

            # 5. Quantiles
            q05, q25, q50, q75, q95 = np.percentile(data, [5, 25, 50, 75, 95])

            # Append in specific order
            stats.extend([mean, std, s_skew, s_kurt, q05, q25, q50, q75, q95, n_nans])
        else:
            # Fallback if column is missing (should not happen based on EDA)
            stats.extend([0.0] * Config.NUM_STATS_PER_SENSOR)

    return np.array(stats, dtype=np.float32)


def generate_spectrogram(df: pd.DataFrame) -> torch.Tensor:
    """
    Converts raw sensor data into a Log-Mel Spectrogram.

    Args:
        df (pd.DataFrame): Dataframe containing sensor readings.

    Returns:
        torch.Tensor: Spectrogram tensor of shape [Channels, n_mels, time].
                      Example: [10, 128, 235]
    """
    # 1. Preprocessing: Fill NaNs with 0
    df_filled = df.fillna(0.0)

    # 2. Convert to Tensor and Transpose to (Channels, Time)
    # Expected input for MelSpectrogram is (..., Time)
    sensor_cols = [f"sensor_{i}" for i in range(1, Config.NUM_SENSORS + 1)]
    waveform = torch.tensor(df_filled[sensor_cols].values, dtype=torch.float32).T

    # 3. Define Transforms
    # Note: In a production pipeline, these might be instantiated once outside,
    # but instantiating here ensures the function is self-contained and stateless.
    mel_transform = T.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
    )

    db_transform = T.AmplitudeToDB(top_db=Config.TOP_DB)

    # 4. Apply Transforms
    spec = mel_transform(waveform)
    spec_db = db_transform(spec)

    return spec_db


def get_statistical_features(
    metadata_path: str, save_path: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads or computes statistical features for a dataset defined by metadata.

    Args:
        metadata_path (str): Path to the metadata CSV (train.csv, val.csv, or test.csv).
        save_path (str): Path where the computed features Parquet file should be saved/loaded.
        load_cached_data (bool): If True, attempts to load from save_path first.

    Returns:
        pd.DataFrame: DataFrame containing 'segment_id' and statistical features.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(save_path):
        print(f"Loading cached features from {save_path}...")
        try:
            return pd.read_parquet(save_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)
    print(f"Computing statistics for {len(df_meta)} files from {metadata_path}...")

    # 3. Compute Features
    feature_list = []

    # Define column names for the output dataframe
    stat_names = [
        "mean",
        "std",
        "skew",
        "kurt",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "nans",
    ]
    col_names = ["segment_id"]
    for i in range(1, Config.NUM_SENSORS + 1):
        for stat in stat_names:
            col_names.append(f"sensor_{i}_{stat}")

    for _, row in df_meta.iterrows():
        segment_id = row["segment_id"]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        if os.path.exists(file_path):
            try:
                # Load raw data
                df_raw = pd.read_csv(file_path)

                # Compute stats
                stats_vec = compute_statistics(df_raw)

                # Prepend segment_id
                row_data = np.concatenate(([segment_id], stats_vec))
                feature_list.append(row_data)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                # Append row of zeros/NaNs or skip?
                # Skipping might desync with metadata, so we append zeros and keep segment_id
                zeros = np.zeros(Config.NUM_STAT_FEATURES)
                row_data = np.concatenate(([segment_id], zeros))
                feature_list.append(row_data)
        else:
            print(f"File not found: {file_path}")
            zeros = np.zeros(Config.NUM_STAT_FEATURES)
            row_data = np.concatenate(([segment_id], zeros))
            feature_list.append(row_data)

    # 4. Create DataFrame
    df_features = pd.DataFrame(feature_list, columns=col_names)

    # Ensure segment_id is int
    df_features["segment_id"] = df_features["segment_id"].astype(int)

    # 5. Save to Cache
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_features.to_parquet(save_path, index=False)
    print(f"Saved features to {save_path}")

    return df_features
