import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torchaudio
from library.config import Config
from library.utils import CacheManager


class FeatureExtractor:
    """
    Extracts tabular features from seismic sensor data, including global statistics,
    spectral properties, and temporal trend features.
    """

    def __init__(self):
        self.cache_manager = CacheManager()
        self.freq_bands = Config.FREQ_BANDS
        self.sample_rate = Config.SAMPLE_RATE
        self.sensors = Config.SENSOR_COLS

    def _compute_fft_bands(self, signal):
        """Computes log energy in specified frequency bands."""
        # Real FFT
        fft_vals = np.fft.rfft(signal)
        fft_freqs = np.fft.rfftfreq(len(signal), d=1 / self.sample_rate)

        # Power spectrum
        power_spectrum = np.abs(fft_vals) ** 2

        features = {}
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            # Find indices for the band
            idx = np.where((fft_freqs >= low_freq) & (fft_freqs < high_freq))[0]
            if len(idx) > 0:
                energy = np.sum(power_spectrum[idx])
                # Log scaling for better distribution
                features[f"log_energy_{band_name}"] = np.log1p(energy)
            else:
                features[f"log_energy_{band_name}"] = 0.0
        return features

    def _compute_mfcc(self, signal):
        """Computes robust statistics for MFCCs."""
        # We use n_mfcc=14 to get coefficients 0-13, then discard 0
        # Hop length and n_fft can be smaller for 1D feature extraction speed

        # Convert to tensor for torchaudio
        sig_tensor = torch.from_numpy(signal).float()

        transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=14,
            melkwargs={"n_fft": 1024, "hop_length": 512},
        )

        mfccs = transform(sig_tensor).numpy()

        # Drop the 0th coefficient (energy)
        mfccs = mfccs[1:, :]

        features = {}
        for i in range(mfccs.shape[0]):
            coeff_idx = i + 1
            # Robust stats: Median, Q05, Q95
            features[f"mfcc_{coeff_idx}_median"] = np.median(mfccs[i])
            features[f"mfcc_{coeff_idx}_q05"] = np.percentile(mfccs[i], 5)
            features[f"mfcc_{coeff_idx}_q95"] = np.percentile(mfccs[i], 95)

        return features

    def _compute_block_metrics(self, signal):
        """
        Computes specific high-energy metrics for a signal block
        to be used in trend calculation.
        """
        metrics = {}

        # 1. Abs Quantile 99 (Proxy for peak amplitude)
        metrics["abs_q99"] = np.percentile(np.abs(signal), 99)

        # 2. Band Energies
        fft_feats = self._compute_fft_bands(signal)
        metrics.update(fft_feats)

        return metrics

    def _extract_sensor_features(self, signal, sensor_name):
        """Extracts all features for a single sensor."""
        features = {}

        # --- Global Time Domain Stats ---
        features[f"{sensor_name}_mean"] = np.mean(signal)
        features[f"{sensor_name}_std"] = np.std(signal)
        features[f"{sensor_name}_min"] = np.min(signal)
        features[f"{sensor_name}_max"] = np.max(signal)
        features[f"{sensor_name}_skew"] = stats.skew(signal)
        features[f"{sensor_name}_kurtosis"] = stats.kurtosis(signal)

        # Quantiles
        q_vals = np.percentile(signal, [1, 5, 95, 99])
        features[f"{sensor_name}_q01"] = q_vals[0]
        features[f"{sensor_name}_q05"] = q_vals[1]
        features[f"{sensor_name}_q95"] = q_vals[2]
        features[f"{sensor_name}_q99"] = q_vals[3]

        # Absolute stats
        abs_sig = np.abs(signal)
        features[f"{sensor_name}_abs_max"] = np.max(abs_sig)
        features[f"{sensor_name}_abs_mean"] = np.mean(abs_sig)
        features[f"{sensor_name}_abs_q99"] = np.percentile(abs_sig, 99)

        # Zero Crossing Rate (Raw count)
        zcr = ((signal[:-1] * signal[1:]) < 0).sum()
        features[f"{sensor_name}_zcr"] = zcr

        # --- Global Frequency Domain ---
        fft_features = self._compute_fft_bands(signal)
        for k, v in fft_features.items():
            features[f"{sensor_name}_{k}"] = v

        # --- MFCCs ---
        mfcc_features = self._compute_mfcc(signal)
        for k, v in mfcc_features.items():
            features[f"{sensor_name}_{k}"] = v

        # --- Trend Features (Segmented) ---
        # Define blocks: Start (first 50%), End (last 50%)
        # Overlap is implicit if we take halves, but let's do thirds overlapping
        # Block 1: 0 to 30000
        # Block 3: 30000 to end
        n_samples = len(signal)
        mid_point = n_samples // 2

        block_start = signal[:mid_point]
        block_end = signal[mid_point:]

        metrics_start = self._compute_block_metrics(block_start)
        metrics_end = self._compute_block_metrics(block_end)

        # Calculate gradients (End - Start)
        for k in metrics_start.keys():
            trend_val = metrics_end[k] - metrics_start[k]
            features[f"{sensor_name}_trend_{k}"] = trend_val

        return features

    def process_segment(self, df_segment):
        """
        Processes a single segment DataFrame (60001 rows x 10 cols)
        and returns a flat feature dictionary.
        """
        # Fill NaNs globally for the segment
        df_segment = df_segment.fillna(df_segment.mean()).fillna(0)

        segment_features = {}

        # 1. Per-Sensor Features
        for sensor in self.sensors:
            if sensor in df_segment.columns:
                # Ensure float32/64
                sig = df_segment[sensor].values.astype(np.float32)
                feats = self._extract_sensor_features(sig, sensor)
                segment_features.update(feats)

        # 2. Spatial Features (Correlation)
        # Compute correlation matrix
        corr_matrix = df_segment[self.sensors].corr().abs()

        # Mean correlation (excluding diagonal)
        mask = np.ones_like(corr_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        mean_corr = corr_matrix.values[mask].mean()
        max_corr = corr_matrix.values[mask].max()

        segment_features["spatial_corr_mean"] = mean_corr
        segment_features["spatial_corr_max"] = max_corr

        return segment_features

    def process_dataset(
        self, metadata_path, output_path, load_cached_data=True, debug_sample=None
    ):
        """
        Generates features for a dataset defined by metadata_path.
        Handles caching and data loading.
        """
        # 1. Check Cache
        if load_cached_data:
            if self.cache_manager.is_cache_valid(output_path):
                print(f"Loading cached features from {output_path}")
                return pd.read_parquet(output_path)
            else:
                print(f"Cache invalid or missing for {output_path}. Regenerating...")

        # 2. Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)

        if debug_sample:
            df_meta = df_meta.head(debug_sample)
            print(f"Debug mode: processing first {debug_sample} rows.")

        # 3. Process Files
        feature_list = []

        print(f"Extracting features for {len(df_meta)} segments...")

        for _, row in df_meta.iterrows():
            seg_id = row[Config.SEGMENT_ID_COL]
            # Construct path: ./input + relative_path_from_metadata
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found. Skipping.")
                continue

            try:
                # Load sensor data
                df_seg = pd.read_csv(file_path, dtype="float32")

                # Extract features
                feats = self.process_segment(df_seg)
                feats[Config.SEGMENT_ID_COL] = seg_id

                # Add target if available
                if Config.TARGET_COL in row:
                    feats[Config.TARGET_COL] = row[Config.TARGET_COL]

                feature_list.append(feats)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        # 4. Create DataFrame
        df_features = pd.DataFrame(feature_list)

        # 5. Save to Cache
        Config.make_dirs()
        df_features.to_parquet(output_path, index=False)
        self.cache_manager.update_cache_metadata(output_path)
        print(f"Features saved to {output_path}")

        return df_features
