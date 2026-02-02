import os
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import torch
import torchaudio
import torchaudio.transforms as T
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    SAMPLING_RATE,
    SAVGOL_WINDOW,
    SAVGOL_POLYORDER,
    N_FFT,
    HOP_LENGTH,
    F_MIN,
    F_MAX,
    IMG_SIZE,
    SEED,
)
from library.utils import seed_everything

# Set seed for reproducibility in signal processing
seed_everything(SEED)


def load_sensor_data(file_path):
    """
    Loads sensor data from a CSV file.
    Fills missing values (NaNs) with the mean of each sensor column.
    """
    full_path = os.path.join(INPUT_DIR, file_path)
    # Load as float32 to save memory and handle potential NaNs
    df = pd.read_csv(full_path, dtype="float32")

    # Impute NaNs with column means
    df = df.fillna(df.mean()).fillna(0)
    return df


def apply_savgol_filter(data):
    """
    Applies Savitzky-Golay smoothing to the data.
    """
    return signal.savgol_filter(
        data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER, axis=0
    )


def compute_spectral_features(x, fs=SAMPLING_RATE):
    """
    Computes spectral features using Welch's method.
    Returns a dictionary of features.
    """
    f, Pxx = signal.welch(x, fs=fs, nperseg=256, axis=0)

    # Spectral Centroid
    # sum(f * Pxx) / sum(Pxx)
    # Handle division by zero if signal is flat
    sum_pxx = np.sum(Pxx)
    if sum_pxx == 0:
        centroid = 0
    else:
        centroid = np.sum(f * Pxx) / sum_pxx

    # Peak Frequency
    peak_freq = f[np.argmax(Pxx)]

    # Band Powers
    # Define bands: 0-5, 5-10, 10-20, 20-50 Hz
    bands = [(0, 5), (5, 10), (10, 20), (20, 50)]
    band_powers = {}

    for low, high in bands:
        idx = np.logical_and(f >= low, f < high)
        band_powers[f"power_{low}_{high}"] = np.sum(Pxx[idx])

    return {"spec_centroid": centroid, "spec_peak_freq": peak_freq, **band_powers}


def extract_tabular_features(df):
    """
    Extracts a comprehensive set of statistical and spectral features from the sensor dataframe.
    Returns a flat dictionary representing a single row in the tabular dataset.
    """
    features = {}
    sensors = df.columns

    # 1. Global Statistics
    # Compute basic stats efficiently
    means = df.mean()
    stds = df.std()
    mins = df.min()
    maxs = df.max()
    skews = df.skew()
    kurts = df.kurtosis()
    quantiles = df.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])

    for sensor in sensors:
        s_data = df[sensor].values

        # Basic Stats
        features[f"{sensor}_mean"] = means[sensor]
        features[f"{sensor}_std"] = stds[sensor]
        features[f"{sensor}_min"] = mins[sensor]
        features[f"{sensor}_max"] = maxs[sensor]
        features[f"{sensor}_skew"] = skews[sensor]
        features[f"{sensor}_kurt"] = kurts[sensor]

        # Quantiles
        features[f"{sensor}_q01"] = quantiles.loc[0.01, sensor]
        features[f"{sensor}_q05"] = quantiles.loc[0.05, sensor]
        features[f"{sensor}_q25"] = quantiles.loc[0.25, sensor]
        features[f"{sensor}_q50"] = quantiles.loc[0.50, sensor]
        features[f"{sensor}_q75"] = quantiles.loc[0.75, sensor]
        features[f"{sensor}_q95"] = quantiles.loc[0.95, sensor]
        features[f"{sensor}_q99"] = quantiles.loc[0.99, sensor]

        # 2. Structural Spectral Features
        spec_feats = compute_spectral_features(s_data)
        for k, v in spec_feats.items():
            features[f"{sensor}_{k}"] = v

        # 3. Windowed Temporal Features (Evolution)
        # Split into 10 non-overlapping windows
        n_windows = 10
        window_size = len(s_data) // n_windows
        # Truncate to fit
        reshaped = s_data[: window_size * n_windows].reshape(n_windows, window_size)

        # Compute local stats
        win_means = np.mean(reshaped, axis=1)
        win_stds = np.std(reshaped, axis=1)

        # Features describing the evolution of the signal
        features[f"{sensor}_win_mean_std"] = np.std(
            win_means
        )  # Variation of the baseline
        features[f"{sensor}_win_std_mean"] = np.mean(win_stds)  # Average noise level
        features[f"{sensor}_win_std_std"] = np.std(
            win_stds
        )  # Variation of the noise level

    return features


def process_tabular_data(metadata_path, save_filename, load_cached_data=True):
    """
    Main function to generate the tabular dataset for LightGBM.
    Implements caching using Parquet.

    Args:
        metadata_path (str): Path to the metadata CSV (train/val/test).
        save_filename (str): Name of the output parquet file (e.g., 'train_features.parquet').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed feature matrix.
    """
    save_path = os.path.join(WORKING_DIR, save_filename)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(save_path):
        print(f"Loading cached tabular features from {save_path}")
        return pd.read_parquet(save_path)

    # 2. Compute from Scratch
    print(f"Generating tabular features for {os.path.basename(metadata_path)}...")
    meta_df = pd.read_csv(metadata_path)

    feature_list = []

    # Iterate over all segments
    for _, row in meta_df.iterrows():
        segment_id = row["segment_id"]
        file_path = row["file_path"]

        # Load and process
        df = load_sensor_data(file_path)

        # Extract features
        feats = extract_tabular_features(df)
        feats["segment_id"] = segment_id

        # Add target if available
        if "time_to_eruption" in row:
            feats["time_to_eruption"] = row["time_to_eruption"]

        feature_list.append(feats)

    # Create DataFrame
    result_df = pd.DataFrame(feature_list)

    # 3. Save Cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    result_df.to_parquet(save_path, index=False)
    print(f"Saved tabular features to {save_path}")

    return result_df


def generate_spectrogram(file_path):
    """
    Generates a Log-Mel Spectrogram for a given sensor file.
    Designed for Stream B (CNN).

    Args:
        file_path (str): Relative path to the sensor CSV.

    Returns:
        torch.Tensor: A tensor of shape (C, H, W) -> (10, 224, 224).
    """
    # Load data
    df = load_sensor_data(file_path)

    # Convert to Tensor: (Time, Channels) -> (Channels, Time)
    waveform = torch.tensor(df.values, dtype=torch.float32).T

    # Define MelSpectrogram transform
    # We use the config parameters
    transform = T.MelSpectrogram(
        sample_rate=SAMPLING_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        f_min=F_MIN,
        f_max=F_MAX,
        n_mels=IMG_SIZE[0],  # Height of the image
        power=2.0,
    )

    # Generate Spectrogram: (Channels, n_mels, time_frames)
    # Note: torchaudio handles multi-channel inputs if shape is (..., time)
    melspec = transform(waveform)

    # Log-Transform (dB scale)
    # Add epsilon for numerical stability
    melspec = 10.0 * torch.log10(melspec + 1e-10)

    # Resize to fixed width (IMG_SIZE[1])
    # melspec shape is (Channels, H, Time_Variable)
    # We need (Channels, H, W)
    # Unsqueeze to add batch dim for interpolate: (1, C, H, Time)
    melspec = melspec.unsqueeze(0)

    # Interpolate
    melspec = torch.nn.functional.interpolate(
        melspec, size=IMG_SIZE, mode="bilinear", align_corners=False
    )

    # Squeeze batch dim back: (C, H, W)
    melspec = melspec.squeeze(0)

    # Normalization
    # Standardize per channel to help CNN convergence
    # Mean and Std over H, W dimensions
    mean = melspec.mean(dim=(1, 2), keepdim=True)
    std = melspec.std(dim=(1, 2), keepdim=True)

    # Avoid division by zero
    melspec = (melspec - mean) / (std + 1e-6)

    return melspec
