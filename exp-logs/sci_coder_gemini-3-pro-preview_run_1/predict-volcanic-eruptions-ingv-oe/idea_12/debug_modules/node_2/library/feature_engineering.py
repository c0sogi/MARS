import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import librosa
from library.config import Config
from library.utils import save_parquet, load_parquet, seed_everything


class TabularFeatureEngineer:
    """
    Implements the feature engineering pipeline for the Tabular Branch (Branch A).
    Extracts Energy-Aware, Impulsiveness, and Robust Spectral features.
    """

    def __init__(self):
        self.fs = Config.SAMPLING_RATE
        self.subband_edges = Config.SUBBAND_EDGES
        self.n_mfcc = Config.MFCC_N_MFCC
        self.n_fft_mfcc = Config.MFCC_N_FFT
        self.hop_length_mfcc = Config.MFCC_HOP_LENGTH
        self.sensors = [f"sensor_{i}" for i in range(1, Config.NUM_SENSORS + 1)]

    def _compute_subband_energy(self, x):
        """
        Calculates Log-Sum-Squared Energy in specific frequency bands.
        """
        # Compute FFT
        n = len(x)
        fft_vals = np.fft.rfft(x)
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)

        # Power Spectrum (magnitude squared)
        power_spectrum = np.abs(fft_vals) ** 2

        features = {}
        for i in range(len(self.subband_edges) - 1):
            low = self.subband_edges[i]
            high = self.subband_edges[i + 1]

            # Find indices corresponding to the band
            idx = np.where((fft_freqs >= low) & (fft_freqs < high))[0]

            if len(idx) > 0:
                # Sum energy in band
                energy = np.sum(power_spectrum[idx])
                # Log transform to handle high dynamic range
                # Add epsilon for numerical stability
                features[f"sb_energy_{low}_{high}"] = np.log(energy + 1e-9)
            else:
                features[f"sb_energy_{low}_{high}"] = 0.0

        return features

    def _compute_impulsiveness(self, x):
        """
        Calculates Crest Factor and Shape Factor to quantify signal impulsiveness.
        """
        abs_x = np.abs(x)
        peak = np.max(abs_x)
        rms = np.sqrt(np.mean(x**2))
        mean_abs = np.mean(abs_x)

        features = {}

        # Crest Factor: Peak / RMS
        if rms > 0:
            features["crest_factor"] = peak / rms
        else:
            features["crest_factor"] = 0.0

        # Shape Factor: RMS / Mean Absolute
        if mean_abs > 0:
            features["shape_factor"] = rms / mean_abs
        else:
            features["shape_factor"] = 0.0

        return features

    def _compute_robust_mfcc(self, x):
        """
        Extracts MFCCs and aggregates them using robust statistics (Median, Q05, Q95).
        """
        # Extract MFCCs
        # x must be float32 or float64. Librosa expects numpy array.
        # Ensure x is contiguous
        x = np.asfortranarray(x)

        try:
            mfccs = librosa.feature.mfcc(
                y=x,
                sr=self.fs,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft_mfcc,
                hop_length=self.hop_length_mfcc,
            )

            # Remove 0th coefficient (energy) if it dominates, but config says 1-13.
            # Usually librosa returns n_mfcc coefficients. We keep all requested.

            features = {}
            for i in range(mfccs.shape[0]):
                coeff = mfccs[i, :]
                features[f"mfcc_{i}_median"] = np.median(coeff)
                features[f"mfcc_{i}_q05"] = np.quantile(coeff, 0.05)
                features[f"mfcc_{i}_q95"] = np.quantile(coeff, 0.95)
                features[f"mfcc_{i}_std"] = np.std(coeff)

            return features
        except Exception:
            # Fallback for very short signals or errors
            features = {}
            for i in range(self.n_mfcc):
                features[f"mfcc_{i}_median"] = 0.0
                features[f"mfcc_{i}_q05"] = 0.0
                features[f"mfcc_{i}_q95"] = 0.0
                features[f"mfcc_{i}_std"] = 0.0
            return features

    def _compute_time_stats(self, x):
        """
        Computes standard and robust time-domain statistics.
        """
        features = {}

        # Basic Stats
        features["mean"] = np.mean(x)
        features["std"] = np.std(x)
        features["min"] = np.min(x)
        features["max"] = np.max(x)
        features["kurtosis"] = stats.kurtosis(x)
        features["skew"] = stats.skew(x)

        # Quantiles
        features["q01"] = np.quantile(x, 0.01)
        features["q05"] = np.quantile(x, 0.05)
        features["q95"] = np.quantile(x, 0.95)
        features["q99"] = np.quantile(x, 0.99)

        # Absolute Quantiles (Energy distribution)
        abs_x = np.abs(x)
        features["abs_q50"] = np.median(abs_x)
        features["abs_q95"] = np.quantile(abs_x, 0.95)
        features["abs_max"] = np.max(abs_x)

        # Zero Crossing Rate (Raw count)
        zcr = ((x[:-1] * x[1:]) < 0).sum()
        features["zcr"] = zcr

        return features

    def _process_segment(self, file_path, segment_id):
        """
        Loads a CSV file and computes all features for all sensors.
        """
        try:
            # Load data
            df = pd.read_csv(file_path, dtype="float32")

            # Fill NaNs with mean per column to maintain DC offset if any,
            # or 0 if preferred. Using fillna(0) is safer for spectral analysis.
            df = df.fillna(0.0)

            row_features = {"segment_id": segment_id}

            # 1. Per-Sensor Features
            sensor_data_dict = {}

            for sensor in self.sensors:
                if sensor in df.columns:
                    x = df[sensor].values
                    sensor_data_dict[sensor] = x

                    # Compute Feature Groups
                    time_feats = self._compute_time_stats(x)
                    imp_feats = self._compute_impulsiveness(x)
                    sb_feats = self._compute_subband_energy(x)
                    mfcc_feats = self._compute_robust_mfcc(x)

                    # Merge and prefix
                    all_feats = {**time_feats, **imp_feats, **sb_feats, **mfcc_feats}
                    for k, v in all_feats.items():
                        row_features[f"{sensor}_{k}"] = v
                else:
                    # Handle missing sensor column if necessary
                    pass

            # 2. Spatial Features (Correlation Matrix)
            # We compute the correlation between all pairs of sensors
            if len(sensor_data_dict) > 1:
                # Stack data: (n_samples, n_sensors)
                data_matrix = np.stack(
                    [
                        sensor_data_dict[s]
                        for s in self.sensors
                        if s in sensor_data_dict
                    ],
                    axis=1,
                )

                # Compute Correlation Matrix
                # Add small noise to avoid constant input division by zero
                if np.std(data_matrix) == 0:
                    corr_matrix = np.eye(data_matrix.shape[1])
                else:
                    corr_matrix = np.corrcoef(data_matrix, rowvar=False)
                    # Handle NaNs in correlation (e.g. constant sensor)
                    corr_matrix = np.nan_to_num(corr_matrix)

                # Extract upper triangle off-diagonals
                k = 0
                active_sensors = [s for s in self.sensors if s in sensor_data_dict]
                for i in range(len(active_sensors)):
                    for j in range(i + 1, len(active_sensors)):
                        s1 = active_sensors[i]
                        s2 = active_sensors[j]
                        row_features[f"corr_{s1}_{s2}"] = corr_matrix[i, j]

            return row_features

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Return minimal dict with segment_id so it doesn't break the dataframe creation
            return {"segment_id": segment_id}

    def create_tabular_dataset(self, metadata_df, subset_name, load_cached_data=True):
        """
        Generates the feature matrix for a given subset.
        Implements strict caching logic.

        Args:
            metadata_df (pd.DataFrame): Metadata with segment_id and file_path.
            subset_name (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: Feature matrix including segment_id.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"{subset_name}_features.parquet")

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"[{subset_name}] Loading tabular features from cache: {cache_path}")
            return load_parquet(cache_path)

        print(
            f"[{subset_name}] Generating tabular features for {len(metadata_df)} segments..."
        )

        # 2. Generate Features
        features_list = []

        for idx, row in metadata_df.iterrows():
            seg_id = row["segment_id"]
            rel_path = row["file_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            if os.path.exists(full_path):
                feats = self._process_segment(full_path, seg_id)
                features_list.append(feats)
            else:
                # If file missing, skip or handle?
                # Metadata verification ensures files exist, but safe to check.
                pass

        # 3. Create DataFrame
        df_features = pd.DataFrame(features_list)

        # Ensure segment_id is int
        if "segment_id" in df_features.columns:
            df_features["segment_id"] = df_features["segment_id"].astype(int)

        # Fill any NaNs generated during feature extraction (e.g. log(0))
        df_features = df_features.fillna(0.0)

        # 4. Save to Cache
        print(
            f"[{subset_name}] Saving {df_features.shape[0]} rows to cache: {cache_path}"
        )
        save_parquet(df_features, cache_path)

        return df_features
