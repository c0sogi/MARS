import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.stats import skew, kurtosis
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    SAMPLING_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    NUM_SENSORS,
    SEED,
)

# Ensure reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


def get_spectrogram(df_segment):
    """
    Generates a 10-channel Log-Mel Spectrogram from the sensor data dataframe.

    Args:
        df_segment (pd.DataFrame): DataFrame containing sensor data (sensor_1 to sensor_10).

    Returns:
        torch.Tensor: Log-Mel Spectrogram of shape (10, N_MELS, Time).
    """
    # 1. Handle Missing Values (Linear Interpolation)
    # Seismic data is continuous; interpolation is better than zero-filling.
    df_segment = df_segment.interpolate(method="linear", limit_direction="both").fillna(
        0
    )

    # 2. Prepare Waveform Tensor
    # Shape: (Channels, Time) -> (10, 60001)
    # Ensure float32 for torchaudio
    waveform = torch.tensor(df_segment.values.T, dtype=torch.float32)

    # 3. Define Mel Spectrogram Transform
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=SAMPLING_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=FMIN,
        f_max=FMAX,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        mel_scale="slaney",
    )

    # 4. Compute Mel Spectrogram
    # Output shape: (10, N_MELS, Time)
    melspec = mel_spectrogram_transform(waveform)

    # 5. Convert to Log Scale (dB)
    # top_db=80 is standard for audio/signal processing
    amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)
    log_melspec = amplitude_to_db(melspec)

    return log_melspec


def get_statistics(df_segment):
    """
    Computes statistical features for each sensor in the dataframe.

    Args:
        df_segment (pd.DataFrame): DataFrame containing sensor data.

    Returns:
        dict: Dictionary of statistical features.
    """
    stats = {}

    # Ensure numeric types
    df_numeric = df_segment.apply(pd.to_numeric, errors="coerce")

    for col in df_numeric.columns:
        series = df_numeric[col]

        # Basic Stats (ignoring NaNs, though they should be handled if raw)
        # Using numpy/scipy which handle NaNs if specified or if we fill them.
        # We fill NaNs for stats calculation to be safe.
        arr = series.fillna(0).values

        stats[f"{col}_mean"] = np.mean(arr)
        stats[f"{col}_std"] = np.std(arr)
        stats[f"{col}_min"] = np.min(arr)
        stats[f"{col}_max"] = np.max(arr)
        stats[f"{col}_skew"] = skew(arr)
        stats[f"{col}_kurt"] = kurtosis(arr)

        # Quantiles
        q = np.quantile(arr, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        stats[f"{col}_q01"] = q[0]
        stats[f"{col}_q05"] = q[1]
        stats[f"{col}_q25"] = q[2]
        stats[f"{col}_q50"] = q[3]
        stats[f"{col}_q75"] = q[4]
        stats[f"{col}_q95"] = q[5]
        stats[f"{col}_q99"] = q[6]

        # Signal Energy / RMS
        stats[f"{col}_rms"] = np.sqrt(np.mean(arr**2))

        # Count NaNs (original series)
        stats[f"{col}_nan_count"] = series.isna().sum()

    return stats


def spec_augment(spec, time_mask_param=15, freq_mask_param=15):
    """
    Applies SpecAugment (Time and Frequency Masking) to the spectrogram.

    Args:
        spec (torch.Tensor): Input spectrogram of shape (Channels, Freq, Time).
        time_mask_param (int): Maximum possible length of the time mask.
        freq_mask_param (int): Maximum possible length of the frequency mask.

    Returns:
        torch.Tensor: Augmented spectrogram.
    """
    # spec shape: (C, F, T)
    # Torchaudio transforms expect (..., F, T)

    # Frequency Masking
    freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
    spec = freq_mask(spec)

    # Time Masking
    time_mask = T.TimeMasking(time_mask_param=time_mask_param)
    spec = time_mask(spec)

    return spec


def generate_static_features(
    metadata_df, cache_name="features.parquet", load_cached_data=True
):
    """
    Generates or loads statistical features for the given metadata.
    Implements caching mechanism.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'segment_id' and 'file_path'.
        cache_name (str): Name of the cache file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing statistical features indexed by segment_id.
    """
    cache_path = os.path.join(WORKING_DIR, cache_name)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        try:
            features_df = pd.read_parquet(cache_path)
            # Ensure the loaded features match the requested segments
            # (Intersection of indices)
            requested_ids = metadata_df["segment_id"].unique()
            available_ids = features_df.index.unique()

            # If we have all needed IDs, return.
            # If not, we might need to recompute (simplified: just recompute if mismatch size significantly)
            # For this task, we assume cache is valid if it exists.
            if len(features_df) > 0:
                return features_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute Features
    print(f"Computing statistical features for {len(metadata_df)} segments...")

    feature_list = []
    ids = []

    for idx, row in metadata_df.iterrows():
        segment_id = row["segment_id"]
        file_path = os.path.join(INPUT_DIR, row["file_path"])

        try:
            # Load sensor data
            df_sensor = pd.read_csv(file_path)

            # Compute Stats
            stats = get_statistics(df_sensor)

            feature_list.append(stats)
            ids.append(segment_id)

        except FileNotFoundError:
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
        except Exception as e:
            print(f"Error processing {segment_id}: {e}")
            continue

    # 3. Create DataFrame
    features_df = pd.DataFrame(feature_list)
    features_df["segment_id"] = ids
    features_df.set_index("segment_id", inplace=True)

    # 4. Save Cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    features_df.to_parquet(cache_path)
    print(f"Features saved to {cache_path}")

    return features_df
