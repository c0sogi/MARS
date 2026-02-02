import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed
from library.config import Config


class FeatureEngineer:
    """
    Handles signal processing, feature extraction, and caching for the Volcano Eruption Prediction task.
    """

    def __init__(self):
        # Initialize Spectrogram Transforms
        # Using Slaney normalization as specified in Idea 8
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            norm=Config.NORM_TYPE,
            mel_scale="slaney" if Config.NORM_TYPE == "slaney" else "htk",
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=Config.TOP_DB)

    def preprocess_signal(self, df):
        """
        Fills NaNs with zeros and returns the signal as a numpy array.

        Args:
            df (pd.DataFrame): Raw sensor data.

        Returns:
            np.ndarray: Processed signal of shape (Time, Channels).
        """
        return df.fillna(0).values

    def get_spectrogram(self, waveform):
        """
        Generates a Log-Mel Spectrogram from the input waveform.

        Args:
            waveform (np.ndarray or torch.Tensor): Input signal.
                Can be (Time, Channels) or (Channels, Time).

        Returns:
            torch.Tensor: Log-Mel Spectrogram of shape (Channels, n_mels, Time).
        """
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)

        # Ensure shape is (Channels, Time) for torchaudio
        # If input is (Time, Channels) [60001, 10], transpose it.
        if (
            waveform.shape[0] == Config.SIGNAL_LENGTH
            and waveform.shape[1] == Config.NUM_SENSORS
        ):
            waveform = waveform.permute(1, 0)

        # Generate Mel Spectrogram
        spec = self.mel_transform(waveform)

        # Convert to dB
        spec = self.db_transform(spec)

        return spec

    def get_time_features(self, signal):
        """
        Computes time-domain statistical features for each sensor.

        Args:
            signal (np.ndarray): Processed signal of shape (Time, Channels).

        Returns:
            dict: Dictionary of features.
        """
        features = {}
        for i in range(Config.NUM_SENSORS):
            s = signal[:, i]
            sensor_name = f"sensor_{i+1}"

            # Basic Stats
            features[f"{sensor_name}_mean"] = np.mean(s)
            features[f"{sensor_name}_std"] = np.std(s)
            features[f"{sensor_name}_min"] = np.min(s)
            features[f"{sensor_name}_max"] = np.max(s)
            features[f"{sensor_name}_skew"] = skew(s)
            features[f"{sensor_name}_kurt"] = kurtosis(s)

            # Quantiles
            q = np.quantile(s, [0.01, 0.05, 0.95, 0.99])
            features[f"{sensor_name}_q01"] = q[0]
            features[f"{sensor_name}_q05"] = q[1]
            features[f"{sensor_name}_q95"] = q[2]
            features[f"{sensor_name}_q99"] = q[3]

        return features

    def get_freq_features(self, signal):
        """
        Computes frequency-domain features (Centroid, Bandwidth, Dominant Freq).

        Args:
            signal (np.ndarray): Processed signal of shape (Time, Channels).

        Returns:
            dict: Dictionary of features.
        """
        features = {}
        for i in range(Config.NUM_SENSORS):
            s = signal[:, i]
            sensor_name = f"sensor_{i+1}"

            # FFT
            fft_vals = np.fft.rfft(s)
            fft_abs = np.abs(fft_vals)
            freqs = np.fft.rfftfreq(len(s), d=1 / Config.SAMPLE_RATE)

            # Spectral Centroid & Bandwidth
            mag_sum = np.sum(fft_abs)
            if mag_sum == 0:
                centroid = 0
                bandwidth = 0
            else:
                centroid = np.sum(freqs * fft_abs) / mag_sum
                bandwidth = np.sqrt(
                    np.sum(((freqs - centroid) ** 2) * fft_abs) / mag_sum
                )

            # Dominant Frequency
            dom_freq = freqs[np.argmax(fft_abs)]

            features[f"{sensor_name}_spec_centroid"] = centroid
            features[f"{sensor_name}_spec_bandwidth"] = bandwidth
            features[f"{sensor_name}_dom_freq"] = dom_freq

        return features

    def _process_single_file(self, row):
        """
        Helper function to process a single file row from metadata.
        Used for parallel processing.
        """
        segment_id = row["segment_id"]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            df = pd.read_csv(file_path)

            # Extract NaN counts before filling (Meta-feature)
            nan_counts = df.isna().sum().to_dict()
            nan_feats = {f"{k}_nan_count": v for k, v in nan_counts.items()}

            # Preprocess signal (Fill NaNs)
            signal = self.preprocess_signal(df)

            # Extract Features
            time_feats = self.get_time_features(signal)
            freq_feats = self.get_freq_features(signal)

            # Combine all features
            combined = {"segment_id": segment_id}
            combined.update(nan_feats)
            combined.update(time_feats)
            combined.update(freq_feats)

            return combined

        except Exception as e:
            print(f"Error processing segment {segment_id} at {file_path}: {e}")
            return None

    def generate_tabular_features(self, metadata_df, save_path, load_cached_data=True):
        """
        Generates or loads tabular features for the given metadata.

        Args:
            metadata_df (pd.DataFrame): Metadata containing 'segment_id' and 'file_path'.
            save_path (str): Path to save/load the Parquet file.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: DataFrame containing extracted features.
        """
        # 1. Try to load from cache
        if load_cached_data and os.path.exists(save_path):
            print(f"Loading cached features from {save_path}")
            return pd.read_parquet(save_path)

        # 2. Compute features if cache miss or force reload
        print(f"Generating features for {len(metadata_df)} segments...")

        # Use joblib for parallel extraction
        # Note: We pass the method _process_single_file.
        # Since mel_transform is a torch module, pickling might be heavy,
        # but for tabular extraction we don't use the spectrogram transform.
        results = Parallel(n_jobs=Config.NUM_WORKERS)(
            delayed(self._process_single_file)(row) for _, row in metadata_df.iterrows()
        )

        # Filter out any failed processings
        results = [r for r in results if r is not None]

        if not results:
            raise RuntimeError("Feature extraction failed for all samples.")

        feat_df = pd.DataFrame(results)

        # 3. Save to cache
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        feat_df.to_parquet(save_path, index=False)
        print(f"Saved features to {save_path}")

        return feat_df
