import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.stats import skew, kurtosis
from library.config import Config


class SeismicFeatureEngineer:
    """
    Handles feature engineering for the seismic eruption prediction task.
    Generates Log-Mel Spectrograms for the CNN branch and statistical features for the MLP branch.
    """

    def __init__(self):
        # Initialize MelSpectrogram transform
        # We use the parameters defined in Config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLING_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        # Epsilon for log transform to avoid log(0)
        self.eps = 1e-6

    def compute_spectrogram(self, df):
        """
        Computes the Log-Mel Spectrogram for the given sensor data.

        Args:
            df (pd.DataFrame): Dataframe containing sensor readings (shape: [60001, 10]).

        Returns:
            torch.Tensor: Log-Mel Spectrogram of shape (10, n_mels, time_steps).
        """
        # Fill NaNs with 0 (baseline for seismic data)
        # Load as float32 to match torch expectations
        data = df.fillna(0).values.astype(np.float32)

        # Transpose to (channels, time) -> (10, 60001)
        data = data.T

        # Convert to tensor
        waveform = torch.tensor(data)

        # Apply MelSpectrogram
        # Output shape: (10, n_mels, time_steps)
        spec = self.mel_transform(waveform)

        # Apply Log transform (Log-Mel)
        # log(S + eps)
        log_spec = torch.log(spec + self.eps)

        return log_spec

    def compute_statistics(self, df):
        """
        Computes statistical features for the MLP branch.

        Args:
            df (pd.DataFrame): Dataframe containing sensor readings.

        Returns:
            dict: Dictionary of calculated features.
        """
        stats = {}
        # Fill NaNs for stats calculation
        df_clean = df.fillna(0)

        # Quantiles to compute
        quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

        for col in df_clean.columns:
            # Skip non-sensor columns if any (though input usually only has sensors)
            if not col.startswith("sensor"):
                continue

            x = df_clean[col].values

            # Basic Stats
            stats[f"{col}_mean"] = float(np.mean(x))
            stats[f"{col}_std"] = float(np.std(x))
            stats[f"{col}_min"] = float(np.min(x))
            stats[f"{col}_max"] = float(np.max(x))

            # Shape Stats
            stats[f"{col}_skew"] = float(skew(x))
            stats[f"{col}_kurt"] = float(kurtosis(x))

            # Quantiles
            q_vals = np.quantile(x, quantiles)
            stats[f"{col}_q01"] = float(q_vals[0])
            stats[f"{col}_q05"] = float(q_vals[1])
            stats[f"{col}_q25"] = float(q_vals[2])
            stats[f"{col}_q50"] = float(q_vals[3])
            stats[f"{col}_q75"] = float(q_vals[4])
            stats[f"{col}_q95"] = float(q_vals[5])
            stats[f"{col}_q99"] = float(q_vals[6])

            # Zero crossings (proxy for frequency)
            # stats[f"{col}_zero_crossings"] = float(((x[:-1] * x[1:]) < 0).sum())

        return stats

    def cache_tabular_features(self, metadata_path, save_path, load_cached_data=True):
        """
        Generates and caches tabular statistical features for a dataset.

        Args:
            metadata_path (str): Path to the metadata CSV (train/val/test).
            save_path (str): Path to save the Parquet file.
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            pd.DataFrame: The DataFrame containing features and segment_ids.
        """
        # Ensure working directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 1. Check Cache
        if load_cached_data and os.path.exists(save_path):
            print(f"Loading cached tabular features from {save_path}")
            return pd.read_parquet(save_path)

        # 2. Compute from Scratch
        print(f"Computing tabular features for {os.path.basename(metadata_path)}...")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)
        feature_rows = []

        # Iterate through all files in metadata
        for _, row in df_meta.iterrows():
            segment_id = row["segment_id"]
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            try:
                df_sensor = pd.read_csv(file_path)

                # Compute features
                seg_stats = self.compute_statistics(df_sensor)
                seg_stats["segment_id"] = segment_id

                # Add target if available (for train/val)
                if "time_to_eruption" in row:
                    seg_stats["time_to_eruption"] = row["time_to_eruption"]

                feature_rows.append(seg_stats)

            except Exception as e:
                print(f"Error processing segment {segment_id}: {e}")
                continue

        # Create DataFrame
        df_features = pd.DataFrame(feature_rows)

        # Save to Parquet
        df_features.to_parquet(save_path, index=False)
        print(f"Saved tabular features to {save_path} ({len(df_features)} rows)")

        return df_features

    def compute_and_cache_spec_stats(
        self, metadata_path, load_cached_data=True, sample_size=1000
    ):
        """
        Computes global Mean and Std for Spectrograms to perform normalization.
        Caches results to Config.SPEC_MEAN_PATH and Config.SPEC_STD_PATH.

        Args:
            metadata_path (str): Path to training metadata.
            load_cached_data (bool): If True, loads from cache.
            sample_size (int): Number of files to sample for statistics calculation.

        Returns:
            tuple: (mean, std)
        """
        mean_path = Config.SPEC_MEAN_PATH
        std_path = Config.SPEC_STD_PATH

        os.makedirs(os.path.dirname(mean_path), exist_ok=True)

        # 1. Check Cache
        if load_cached_data and os.path.exists(mean_path) and os.path.exists(std_path):
            print("Loading cached spectrogram statistics...")
            spec_mean = np.load(mean_path)
            spec_std = np.load(std_path)
            return spec_mean, spec_std

        # 2. Compute from Scratch
        print("Computing global spectrogram statistics (this may take a moment)...")

        df_meta = pd.read_csv(metadata_path)

        # Sample files to save time
        if len(df_meta) > sample_size:
            df_sample = df_meta.sample(n=sample_size, random_state=Config.SEED)
        else:
            df_sample = df_meta

        sum_spec = 0.0
        sum_sq_spec = 0.0
        count = 0

        for _, row in df_sample.iterrows():
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            try:
                df_sensor = pd.read_csv(file_path)

                # Compute spectrogram
                # Shape: (10, n_mels, time)
                spec = self.compute_spectrogram(df_sensor)

                # Accumulate stats
                # We normalize globally (scalar mean/std) or per-channel?
                # Usually per-channel is better if sensors are different, but here they are similar types.
                # Let's do global scalar mean/std for simplicity and robustness,
                # or we can do per-frequency band.
                # Given the task, a simple global scalar normalization is often sufficient for CNNs.
                # Let's compute scalar stats.

                spec_np = spec.numpy()
                sum_spec += np.sum(spec_np)
                sum_sq_spec += np.sum(spec_np**2)
                count += spec_np.size

            except Exception as e:
                continue

        if count == 0:
            raise ValueError("No data processed for spectrogram stats.")

        # Calculate Mean and Std
        global_mean = sum_spec / count
        global_std = np.sqrt((sum_sq_spec / count) - (global_mean**2))

        # Save
        np.save(mean_path, np.array(global_mean))
        np.save(std_path, np.array(global_std))

        print(
            f"Computed Spectrogram Stats - Mean: {global_mean:.4f}, Std: {global_std:.4f}"
        )
        return global_mean, global_std

    def fit_and_cache_tabular_scaler(self, train_features_path, load_cached_data=True):
        """
        Fits a Standard Scaler on the training tabular features and caches the parameters.

        Args:
            train_features_path (str): Path to the training features parquet file.
            load_cached_data (bool): If True, checks if scaler files exist.
        """
        mean_path = Config.STATS_SCALER_MEAN_PATH
        scale_path = Config.STATS_SCALER_SCALE_PATH

        os.makedirs(os.path.dirname(mean_path), exist_ok=True)

        if (
            load_cached_data
            and os.path.exists(mean_path)
            and os.path.exists(scale_path)
        ):
            print("Tabular scaler already cached.")
            return

        print("Fitting tabular feature scaler...")
        if not os.path.exists(train_features_path):
            raise FileNotFoundError(
                "Train features file not found. Run cache_tabular_features first."
            )

        df_train = pd.read_parquet(train_features_path)

        # Drop non-feature columns
        drop_cols = ["segment_id", "time_to_eruption"]
        feature_cols = [c for c in df_train.columns if c not in drop_cols]

        X = df_train[feature_cols].values

        # Compute Mean and Scale (Std)
        means = np.mean(X, axis=0)
        scales = np.std(X, axis=0)

        # Handle zero variance
        scales[scales == 0] = 1.0

        np.save(mean_path, means)
        np.save(scale_path, scales)

        print(f"Saved tabular scaler to {mean_path} and {scale_path}")
