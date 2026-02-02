import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.stats import skew, kurtosis
from library.config import Config
from library.utils import TargetScaler

# Ensure deterministic behavior
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


def compute_spectrogram(waveform, sr=Config.SAMPLING_RATE):
    """
    Generates a Log-Mel Spectrogram from the raw waveform.

    Args:
        waveform (torch.Tensor): Input signal of shape (Channels, Time).
        sr (int): Sampling rate.

    Returns:
        torch.Tensor: Log-Mel Spectrogram of shape (Channels, n_mels, time_steps).
    """
    # Define Mel Spectrogram transform with Slaney normalization
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=Config.N_FFT,
        win_length=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        norm="slaney",
        mel_scale="slaney",
    )

    # Define Amplitude to DB transform
    db_transform = T.AmplitudeToDB(top_db=Config.TOP_DB)

    # Apply transforms
    # Ensure waveform is float32
    spec = mel_transform(waveform.float())
    spec_db = db_transform(spec)

    return spec_db


def compute_spectral_features(signal, sr=Config.SAMPLING_RATE):
    """
    Computes frequency-domain statistics for a single sensor signal.

    Args:
        signal (np.ndarray): 1D array of sensor readings.
        sr (int): Sampling rate.

    Returns:
        tuple: (centroid, bandwidth, dominant_frequency)
    """
    # Compute FFT
    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(len(signal), d=1 / sr)

    # Normalize magnitude spectrum to treat as a probability distribution
    mag_sum = np.sum(fft_mag)
    if mag_sum == 0:
        return 0.0, 0.0, 0.0

    p = fft_mag / mag_sum

    # Spectral Centroid
    centroid = np.sum(freqs * p)

    # Spectral Bandwidth
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * p))

    # Dominant Frequency
    dom_freq = freqs[np.argmax(fft_mag)]

    return centroid, bandwidth, dom_freq


def process_segment(file_path, return_spec=True):
    """
    Orchestrates the processing of a single CSV segment.
    Reads the file, handles NaNs, and computes both spectrograms and tabular stats.

    Args:
        file_path (str): Path to the CSV file.
        return_spec (bool): If True, returns the spectrogram tensor.
                            If False, returns None for spectrogram (faster for tabular-only generation).

    Returns:
        tuple: (spectrogram_tensor, stats_dict)
    """
    # 1. Read CSV
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Fallback for robustness, though metadata checks should prevent this
        df = pd.DataFrame(
            np.zeros((60001, Config.NUM_SENSORS)),
            columns=[f"sensor_{i}" for i in range(1, 11)],
        )

    # 2. Compute Meta-Features (NaN counts) on raw data
    nan_counts = df.isna().sum()

    # 3. Fill NaNs with 0 (no interpolation)
    df_filled = df.fillna(0)

    # 4. Convert to Numpy for efficient Stats calculation
    data_np = df_filled.values  # Shape: (Time, Channels)

    stats = {}

    # Iterate over each sensor to compute statistics
    for i, col in enumerate(df.columns):
        if i >= Config.NUM_SENSORS:
            break

        sensor_name = f"sensor_{i+1}"
        signal = data_np[:, i]

        # --- Time-Domain Stats ---
        stats[f"{sensor_name}_mean"] = np.mean(signal)
        stats[f"{sensor_name}_std"] = np.std(signal)
        stats[f"{sensor_name}_min"] = np.min(signal)
        stats[f"{sensor_name}_max"] = np.max(signal)
        stats[f"{sensor_name}_skew"] = skew(signal)
        stats[f"{sensor_name}_kurt"] = kurtosis(signal)

        # Quantiles
        q = np.quantile(signal, [0.05, 0.25, 0.50, 0.75, 0.95])
        stats[f"{sensor_name}_q05"] = q[0]
        stats[f"{sensor_name}_q25"] = q[1]
        stats[f"{sensor_name}_q50"] = q[2]
        stats[f"{sensor_name}_q75"] = q[3]
        stats[f"{sensor_name}_q95"] = q[4]

        # NaN Count
        stats[f"{sensor_name}_nan"] = nan_counts.iloc[i]

        # --- Frequency-Domain Stats ---
        cent, bw, dom = compute_spectral_features(signal)
        stats[f"{sensor_name}_spec_cent"] = cent
        stats[f"{sensor_name}_spec_bw"] = bw
        stats[f"{sensor_name}_dom_freq"] = dom

    # 5. Compute Spectrogram (if requested)
    spec_tensor = None
    if return_spec:
        # Transpose to (Channels, Time) for Torchaudio
        waveform = torch.tensor(data_np.T, dtype=torch.float32)
        spec_tensor = compute_spectrogram(waveform)

    return spec_tensor, stats


class FeatureEngineer:
    """
    Manages the generation, scaling, and caching of the tabular feature set.
    """

    def __init__(self):
        self.stats_mean = None
        self.stats_scale = None

    def _process_subset(self, metadata_path, desc="Processing"):
        """
        Helper to process a list of files defined in a metadata CSV.
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)

        # Debugging hook
        if Config.DEBUG:
            df_meta = df_meta.head(Config.DEBUG_SIZE)

        features_list = []
        targets = []
        ids = []

        for idx, row in df_meta.iterrows():
            # Construct full path (metadata paths are relative to input dir)
            full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            # We only need tabular stats here
            _, stats = process_segment(full_path, return_spec=False)

            features_list.append(stats)
            ids.append(row["segment_id"])

            if "time_to_eruption" in row:
                targets.append(row["time_to_eruption"])

        df_features = pd.DataFrame(features_list)
        df_features["segment_id"] = ids

        if targets:
            df_features["target"] = targets

        return df_features

    def run(self, load_cached_data=True):
        """
        Main execution method. Checks for cached parquet files.
        If missing or forced, generates features from scratch, fits scalers, and saves artifacts.
        """
        # 1. Check Cache
        if load_cached_data:
            if (
                os.path.exists(Config.TRAIN_FEATURES_PATH)
                and os.path.exists(Config.VAL_FEATURES_PATH)
                and os.path.exists(Config.TEST_FEATURES_PATH)
            ):
                print("Loading features from cache...")
                return

        print("Generating features from scratch...")

        # 2. Process Train
        print("Processing Train...")
        train_df = self._process_subset(Config.TRAIN_METADATA, "Train")

        # 3. Process Val
        print("Processing Val...")
        val_df = self._process_subset(Config.VAL_METADATA, "Val")

        # 4. Process Test
        print("Processing Test...")
        test_df = self._process_subset(Config.TEST_METADATA, "Test")

        # 5. Fit Target Scaler (on Train Target)
        print("Fitting Target Scaler...")
        target_scaler = TargetScaler()
        target_scaler.fit(train_df["target"].values)
        target_scaler.save(Config.TARGET_MEAN_PATH, Config.TARGET_STD_PATH)

        # 6. Fit Feature Scaler (on Train Features, exclude ID and Target)
        print("Fitting Feature Scaler...")
        feature_cols = [
            c for c in train_df.columns if c not in ["segment_id", "target"]
        ]

        X_train = train_df[feature_cols].values
        self.stats_mean = np.mean(X_train, axis=0)
        self.stats_scale = np.std(X_train, axis=0)

        # Prevent division by zero
        self.stats_scale[self.stats_scale == 0] = 1.0

        # Save Scaler Stats
        np.save(Config.STATS_SCALER_MEAN_PATH, self.stats_mean)
        np.save(Config.STATS_SCALER_SCALE_PATH, self.stats_scale)

        # 7. Transform and Save
        def transform_and_save(df, path):
            X = df[feature_cols].values
            X_scaled = (X - self.stats_mean) / self.stats_scale

            df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
            df_scaled["segment_id"] = df["segment_id"]
            if "target" in df.columns:
                df_scaled["target"] = df["target"]

            df_scaled.to_parquet(path, index=False)

        transform_and_save(train_df, Config.TRAIN_FEATURES_PATH)
        transform_and_save(val_df, Config.VAL_FEATURES_PATH)
        transform_and_save(test_df, Config.TEST_FEATURES_PATH)

        print("Feature generation complete.")
