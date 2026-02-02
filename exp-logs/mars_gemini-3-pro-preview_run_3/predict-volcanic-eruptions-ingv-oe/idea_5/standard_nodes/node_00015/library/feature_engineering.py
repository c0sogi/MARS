import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from joblib import Parallel, delayed
from scipy.fft import rfft, rfftfreq
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from library.config import (
    SENSOR_COLS,
    N_TIME_SEGMENTS,
    ROLLING_WINDOW_SIZE,
    FREQ_BANDS,
    MFCC_PARAMS,
    SAMPLING_RATE,
    WORKING_DIR,
    RANDOM_SEED,
    RFE_PARAMS,
    RFE_TRAIN_SUBSET_SIZE,
    TARGET_COL,
)
from library.data_loader import load_sensor_segment

# Set seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class FeatureExtractor:
    """
    Handles the extraction of signal processing features from sensor data.
    Implements Cepstral (MFCC), Spectral, and Temporal (RMS) analysis
    with spatial aggregation across sensors.
    """

    def __init__(self, cache_dir=WORKING_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize MFCC transform
        # We use CPU for stability in multiprocessing
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=SAMPLING_RATE,
            n_mfcc=MFCC_PARAMS["n_mfcc"],
            melkwargs={
                "n_fft": MFCC_PARAMS["n_fft"],
                "n_mels": MFCC_PARAMS["n_mels"],
                "hop_length": MFCC_PARAMS["hop_length"],
                "center": False,
            },
        )

    def _compute_spatial_stats(self, values, prefix):
        """
        Aggregates feature values across the 10 sensors.
        Returns a dictionary with Mean, Std, Max, Min.
        """
        # values shape: (n_sensors,)
        return {
            f"{prefix}_mean": np.mean(values),
            f"{prefix}_std": np.std(values),
            f"{prefix}_max": np.max(values),
            f"{prefix}_min": np.min(values),
        }

    def _extract_raw_stats(self, df):
        """
        Extracts basic time-domain statistics for each sensor individually.
        """
        features = {}
        for col in df.columns:
            # Calculate basic stats for the raw time series of this sensor
            features[f"{col}_mean"] = df[col].mean()
            features[f"{col}_std"] = df[col].std()
            features[f"{col}_min"] = df[col].min()
            features[f"{col}_max"] = df[col].max()
        return features

    def _extract_mfcc(self, waveform_tensor):
        """
        Extracts MFCC features.
        waveform_tensor: (n_sensors, n_samples)
        """
        # Output shape: (n_sensors, n_mfcc, n_frames)
        mfcc = self.mfcc_transform(waveform_tensor)

        # Statistics over time: Mean and Std
        # Shape: (n_sensors, n_mfcc)
        mfcc_mean = torch.mean(mfcc, dim=2).numpy()
        mfcc_std = torch.std(mfcc, dim=2).numpy()

        features = {}
        n_sensors, n_coeffs = mfcc_mean.shape

        for i in range(n_coeffs):
            # Spatial aggregation for each coefficient's mean
            features.update(self._compute_spatial_stats(mfcc_mean[:, i], f"mfcc_{i}"))
            # Spatial aggregation for each coefficient's std
            features.update(
                self._compute_spatial_stats(mfcc_std[:, i], f"mfcc_{i}_std")
            )

        return features

    def _extract_spectral(self, sensor_data):
        """
        Extracts Band Powers and Spectral Centroid.
        sensor_data: (n_samples, n_sensors) numpy array
        """
        n_samples, n_sensors = sensor_data.shape

        # Compute Real FFT
        # Shape: (n_freqs, n_sensors)
        fft_vals = np.abs(rfft(sensor_data, axis=0))
        freqs = rfftfreq(n_samples, d=1 / SAMPLING_RATE)

        features = {}

        # 1. Band Power
        for band_name, (low_f, high_f) in FREQ_BANDS.items():
            idx = np.where((freqs >= low_f) & (freqs <= high_f))[0]
            if len(idx) > 0:
                # Sum energy in band
                band_power = np.sum(fft_vals[idx, :], axis=0)
                features.update(
                    self._compute_spatial_stats(band_power, f"band_{band_name}")
                )
            else:
                features.update(
                    self._compute_spatial_stats(
                        np.zeros(n_sensors), f"band_{band_name}"
                    )
                )

        # 2. Spectral Centroid
        # sum(f * mag) / sum(mag)
        mag_sum = np.sum(fft_vals, axis=0)
        # Avoid division by zero
        mag_sum[mag_sum == 0] = 1e-10
        centroid = np.sum(freqs[:, None] * fft_vals, axis=0) / mag_sum
        features.update(self._compute_spatial_stats(centroid, "spec_centroid"))

        return features

    def _extract_temporal_windows(self, df):
        """
        Computes Rolling RMS and aggregates over N explicit time segments.
        df: DataFrame with sensor columns
        """
        # 1. Compute Rolling RMS Energy Envelope
        # x^2 -> rolling mean -> sqrt
        # This gives us the energy trajectory of the signal
        energy_envelope = (
            df.pow(2).rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean().pow(0.5)
        )

        # 2. Split into N non-overlapping segments
        # We split the index array to get ranges
        indices = np.array_split(energy_envelope.index, N_TIME_SEGMENTS)

        features = {}

        for i, idx_range in enumerate(indices):
            if len(idx_range) == 0:
                continue

            segment_data = energy_envelope.loc[
                idx_range
            ].values  # (n_samples_in_seg, n_sensors)

            # Compute stats for this window per sensor
            # Mean RMS (Average Energy in this window)
            seg_mean = np.mean(segment_data, axis=0)
            features.update(self._compute_spatial_stats(seg_mean, f"win_{i}_rms_mean"))

            # Max RMS (Peak Energy in this window)
            seg_max = np.max(segment_data, axis=0)
            features.update(self._compute_spatial_stats(seg_max, f"win_{i}_rms_max"))

        return features

    def process_single_segment(self, segment_id, file_path):
        """
        Loads and processes a single segment file.
        """
        try:
            # Load Data (Imputed)
            df = load_sensor_segment(file_path, fill_na=True)

            # 1. Raw Stats (Per Sensor)
            feats = self._extract_raw_stats(df)

            # 2. Temporal Window Features (RMS)
            feats.update(self._extract_temporal_windows(df))

            # 3. Spectral Features
            data_np = df.values
            feats.update(self._extract_spectral(data_np))

            # 3. MFCC Features
            # Convert to tensor (n_sensors, n_samples) for torchaudio
            # Note: df.values is (n_samples, n_sensors), so we transpose
            data_tensor = torch.tensor(data_np.T, dtype=torch.float32)
            feats.update(self._extract_mfcc(data_tensor))

            feats["segment_id"] = int(segment_id)
            return feats

        except Exception as e:
            print(f"Error processing segment {segment_id}: {e}")
            return None

    def generate_features(self, metadata_df, split_name, load_cached_data=True):
        """
        Generates features for all segments in the metadata.
        Uses caching to avoid re-computation.
        """
        cache_path = os.path.join(self.cache_dir, f"{split_name}_features.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split_name} features from cache: {cache_path}")
            df = pd.read_parquet(cache_path)

            # Cite debug_lesson_1: Validate Cached Artifacts Against Source Metadata
            # If the cache was polluted with the target column, remove it to prevent merge collisions (x/y suffixes)
            if TARGET_COL in df.columns:
                print(f"Removing {TARGET_COL} from cached features.")
                df = df.drop(columns=[TARGET_COL])

            return df

        print(
            f"Generating {split_name} features from scratch ({len(metadata_df)} files)..."
        )

        # Parallel Execution
        # We use joblib to parallelize the file processing
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(self.process_single_segment)(row["segment_id"], row["file_path"])
            for _, row in metadata_df.iterrows()
        )

        # Filter out failed processings
        results = [r for r in results if r is not None]

        feat_df = pd.DataFrame(results)

        # Save to cache
        feat_df.to_parquet(cache_path, index=False)
        print(f"Saved {split_name} features to {cache_path}. Shape: {feat_df.shape}")

        return feat_df


class FeatureSelector:
    """
    Wrapper for Recursive Feature Elimination (RFE).
    Selects the most important features to reduce dimensionality and overfitting.
    """

    def __init__(self, cache_dir=WORKING_DIR):
        self.cache_dir = cache_dir
        self.selected_columns_path = os.path.join(cache_dir, "selected_features.txt")
        self.selected_features = None

    def fit(self, X, y):
        """
        Fits RFE on a subset of the data and saves the selected feature names.
        """
        print("Starting Feature Selection (RFE)...")

        # Subsample data for RFE to save time
        n_samples = int(len(X) * RFE_TRAIN_SUBSET_SIZE)
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sub = X.iloc[indices]
        y_sub = y.iloc[indices]

        print(f"RFE fitting on {len(X_sub)} samples...")

        estimator = RandomForestRegressor(**RFE_PARAMS["estimator_params"])
        selector = RFE(
            estimator=estimator,
            n_features_to_select=RFE_PARAMS["n_features_to_select"],
            step=RFE_PARAMS["step"],
            verbose=RFE_PARAMS["verbose"],
        )

        selector.fit(X_sub, y_sub)

        self.selected_features = X.columns[selector.support_].tolist()
        print(f"RFE selected {len(self.selected_features)} features.")

        # Save selected features
        with open(self.selected_columns_path, "w") as f:
            for feat in self.selected_features:
                f.write(f"{feat}\n")

        return self

    def transform(self, X):
        """
        Filters the dataframe to keep only selected features.
        """
        if self.selected_features is None:
            # Try loading from file
            if os.path.exists(self.selected_columns_path):
                with open(self.selected_columns_path, "r") as f:
                    self.selected_features = [line.strip() for line in f]
            else:
                print(
                    "Warning: No selected features found. Returning original dataframe."
                )
                return X

        # Ensure all selected features exist in X
        missing = [c for c in self.selected_features if c not in X.columns]
        if missing:
            print(
                f"Warning: {len(missing)} selected features missing in input. Filling with 0."
            )
            for c in missing:
                X[c] = 0.0

        return X[self.selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def extract_features(metadata_df, split_name, load_cached_data=True):
    """
    Convenience function to run the FeatureExtractor.
    """
    extractor = FeatureExtractor()
    return extractor.generate_features(metadata_df, split_name, load_cached_data)
