import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.stats import skew, kurtosis
from library.config import Config
from library.utils import seed_everything


def load_and_clean_signal(file_path):
    """
    Reads a sensor CSV file and fills NaN values with zeros.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The cleaned sensor data.
    """
    # Read CSV
    df = pd.read_csv(file_path)

    # Fill NaNs with zeros as per "Idea" description
    df = df.fillna(0)

    return df


def extract_statistics(df):
    """
    Computes a vector of statistical features for the MLP branch.

    Args:
        df (pd.DataFrame): The raw sensor data (can contain NaNs).

    Returns:
        pd.Series: A Series containing the computed statistics for all sensors.
    """
    features = {}
    quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

    for i in range(1, Config.NUM_SENSORS + 1):
        col_name = f"sensor_{i}"
        if col_name in df.columns:
            series = df[col_name]

            # NaN Count (Meta-feature)
            features[f"{col_name}_nan_count"] = series.isna().sum()

            # Drop NaNs for statistical calculation
            clean_series = series.dropna()

            if len(clean_series) == 0:
                # Handle empty series if all were NaNs
                features[f"{col_name}_mean"] = 0
                features[f"{col_name}_std"] = 0
                features[f"{col_name}_min"] = 0
                features[f"{col_name}_max"] = 0
                features[f"{col_name}_skew"] = 0
                features[f"{col_name}_kurtosis"] = 0
                for q in quantiles:
                    features[f"{col_name}_q{int(q*100)}"] = 0
            else:
                features[f"{col_name}_mean"] = clean_series.mean()
                features[f"{col_name}_std"] = clean_series.std()
                features[f"{col_name}_min"] = clean_series.min()
                features[f"{col_name}_max"] = clean_series.max()

                # Skew and Kurtosis
                # bias=False corresponds to calculations similar to pandas/scipy default
                features[f"{col_name}_skew"] = skew(clean_series, bias=False)
                features[f"{col_name}_kurtosis"] = kurtosis(clean_series, bias=False)

                # Quantiles
                q_vals = clean_series.quantile(quantiles)
                for q, val in zip(quantiles, q_vals):
                    features[f"{col_name}_q{int(q*100)}"] = val

    return pd.Series(features)


def generate_dual_spectrograms(signal_data):
    """
    Generates two Log-Mel Spectrograms per sensor (Short and Long windows)
    and stacks them along the channel dimension.

    Args:
        signal_data (pd.DataFrame or np.ndarray): The cleaned sensor data.
                                                  Shape (60001, 10).

    Returns:
        torch.Tensor: Stacked Spectrogram with shape (20, 128, Time).
    """
    if isinstance(signal_data, pd.DataFrame):
        signal_data = signal_data.values

    # Convert to Tensor and transpose to (Channels, Time) -> (10, 60001)
    waveform = torch.tensor(signal_data, dtype=torch.float32).T

    # Define Transforms
    # View A: Short Window (High Temporal Resolution)
    spec_transform_short = T.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT_SHORT,
        win_length=Config.WIN_LENGTH_SHORT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        mel_scale="slaney",
    )

    # View B: Long Window (High Frequency Resolution)
    spec_transform_long = T.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT_LONG,
        win_length=Config.WIN_LENGTH_LONG,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        mel_scale="slaney",
    )

    # Amplitude to DB
    db_transform = T.AmplitudeToDB(top_db=Config.TOP_DB)

    # Generate Spectrograms
    # Input: (10, Time) -> Output: (10, n_mels, Time)
    spec_short = db_transform(spec_transform_short(waveform))
    spec_long = db_transform(spec_transform_long(waveform))

    # Stack along channel dimension -> (20, n_mels, Time)
    return torch.cat([spec_short, spec_long], dim=0)


class SpecAugment(torch.nn.Module):
    """
    Applies Time and Frequency Masking to spectrograms.
    """

    def __init__(self, freq_mask_param=10, time_mask_param=30):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)

    def forward(self, spec):
        # spec: (Channels, Freq, Time)
        return self.time_mask(self.freq_mask(spec))


def process_and_cache_features(metadata_path, output_path, load_cached_data=True):
    """
    Orchestrates the extraction of statistical features for a dataset split.
    Caches the result to a Parquet file.

    Args:
        metadata_path (str): Path to the metadata CSV (train.csv, val.csv, or test.csv).
        output_path (str): Path where the parquet file should be saved.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: The DataFrame containing segment_id and engineered features.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached features from {output_path}")
        return pd.read_parquet(output_path)

    print(f"Processing features for {metadata_path}...")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file {metadata_path} not found.")

    df_meta = pd.read_csv(metadata_path)

    # 3. Iterate and Extract
    feature_rows = []

    # Ensure input directory is correct
    input_dir = Config.INPUT_DIR

    for idx, row in df_meta.iterrows():
        segment_id = row["segment_id"]
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        if not os.path.exists(full_path):
            # Should not happen if metadata is correct, but safe to handle
            print(f"Warning: File {full_path} not found. Skipping.")
            continue

        # Read RAW data (to preserve NaNs for counting)
        df_raw = pd.read_csv(full_path)

        # Extract Statistics
        stats = extract_statistics(df_raw)
        stats["segment_id"] = segment_id

        feature_rows.append(stats)

    # 4. Create DataFrame
    df_features = pd.DataFrame(feature_rows)

    # 5. Save to Cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    print(f"Saved features to {output_path}")

    return df_features


def cache_all_features(load_cached_data=True):
    """
    Helper function to process train, val, and test features.
    """
    # Train
    process_and_cache_features(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_FEATURES_PATH, load_cached_data
    )

    # Val
    process_and_cache_features(
        Config.VAL_METADATA_PATH, Config.VAL_FEATURES_PATH, load_cached_data
    )

    # Test
    process_and_cache_features(
        Config.TEST_METADATA_PATH, Config.TEST_FEATURES_PATH, load_cached_data
    )
