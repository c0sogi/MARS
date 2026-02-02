import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import librosa
import warnings

from library.config import (
    INPUT_DIR,
    SENSOR_COLS,
    NUM_SENSORS,
    SAMPLING_RATE,
    MFCC_N_MFCC,
    MFCC_N_FFT,
    MFCC_HOP_LENGTH,
    MFCC_N_MELS,
    SEED,
)
from library.utils import CacheManager, seed_everything

# Suppress librosa warnings that might occur with short signals or specific formats
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TabularFeatureExtractor:
    """
    Implements the feature extraction logic for the Source-Aware Tabular Regressor (Branch A).
    Generates features from physical sensors and a beamformed virtual sensor.
    """

    def __init__(self, cache_dir="./working/idea_7"):
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        seed_everything(SEED)

    def _calculate_beamformed_sensor(self, df):
        """
        Generates a 'Virtual Source' channel by averaging the normalized waveforms
        of all 10 sensors. This acts as a beamformer to amplify common-mode signal.
        """
        # Normalize each sensor (z-score) to ensure equal contribution
        # Handle potential constant signals (std=0) by adding epsilon
        epsilon = 1e-8
        normalized_df = (df[SENSOR_COLS] - df[SENSOR_COLS].mean()) / (
            df[SENSOR_COLS].std() + epsilon
        )

        # Compute the mean across sensors for each time step
        virtual_sensor = normalized_df.mean(axis=1)
        return virtual_sensor

    def _extract_series_features(self, series, prefix):
        """
        Extracts statistical, structural, and cepstral features for a single time series.
        """
        x = series.values
        # Handle NaNs if any remain (though we fill before calling this)
        if np.isnan(x).any():
            x = np.nan_to_num(x)

        features = {}

        # 1. Global Statistics
        features[f"{prefix}_mean"] = np.mean(x)
        features[f"{prefix}_std"] = np.std(x)
        features[f"{prefix}_skew"] = float(stats.skew(x))
        features[f"{prefix}_kurtosis"] = float(stats.kurtosis(x))

        # Quantiles
        quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        q_vals = np.quantile(x, quantiles)
        for q, val in zip(quantiles, q_vals):
            q_str = int(q * 100)
            features[f"{prefix}_q{q_str:02d}"] = val

        # 2. Absolute Statistics (Energy proxies)
        abs_x = np.abs(x)
        features[f"{prefix}_abs_mean"] = np.mean(abs_x)
        features[f"{prefix}_abs_std"] = np.std(abs_x)
        features[f"{prefix}_abs_max"] = np.max(abs_x)

        abs_q_vals = np.quantile(abs_x, [0.95, 0.99])
        features[f"{prefix}_abs_q95"] = abs_q_vals[0]
        features[f"{prefix}_abs_q99"] = abs_q_vals[1]

        # 3. Structural Features: Raw Zero-Crossing Rate
        # We do not center the signal; we want to know how often it crosses 0 amplitude.
        # This acts as a frequency proxy and noise gate.
        zcr = ((x[:-1] * x[1:]) < 0).sum()
        features[f"{prefix}_zcr"] = zcr

        # 4. Parsimonious Cepstral Features (MFCCs)
        # Extract coefficients 1-13 (Low-Order)
        # We ensure the input is float32 for librosa
        mfccs = librosa.feature.mfcc(
            y=x.astype(np.float32),
            sr=SAMPLING_RATE,
            n_mfcc=MFCC_N_MFCC,
            n_fft=MFCC_N_FFT,
            hop_length=MFCC_HOP_LENGTH,
            n_mels=MFCC_N_MELS,
            fmax=None,  # Nyquist
        )

        # Aggregate MFCCs using Robust Statistics only
        # mfccs shape: (n_mfcc, t)
        for i in range(MFCC_N_MFCC):
            coeff = mfccs[i, :]
            features[f"{prefix}_mfcc{i}_mean"] = np.mean(coeff)
            features[f"{prefix}_mfcc{i}_std"] = np.std(coeff)
            features[f"{prefix}_mfcc{i}_q05"] = np.quantile(coeff, 0.05)
            features[f"{prefix}_mfcc{i}_q95"] = np.quantile(coeff, 0.95)

        return features

    def _extract_spatial_features(self, df):
        """
        Computes the Correlation Matrix between all physical sensors.
        """
        features = {}
        # Compute correlation matrix
        corr_matrix = df[SENSOR_COLS].corr()

        # Extract upper triangle elements to avoid duplicates and self-correlation
        for i in range(len(SENSOR_COLS)):
            for j in range(i + 1, len(SENSOR_COLS)):
                col_i = SENSOR_COLS[i]
                col_j = SENSOR_COLS[j]
                # Key format: corr_s1_s2
                # Simplify key to use sensor indices
                idx_i = i + 1
                idx_j = j + 1
                val = corr_matrix.loc[col_i, col_j]
                features[f"corr_s{idx_i}_s{idx_j}"] = val

        return features

    def process_segment(self, segment_id, file_path):
        """
        Reads a CSV file, performs beamforming, and extracts features.
        """
        full_path = os.path.join(INPUT_DIR, file_path)

        # Load data
        try:
            # Use float32 to handle potential nulls and memory
            df = pd.read_csv(full_path, dtype="float32")
        except FileNotFoundError:
            # Should not happen given metadata checks, but robust handling
            print(f"Warning: File {full_path} not found. Returning zeros.")
            return None

        # Fill NaNs - Sensor 2 is known to have NaNs.
        # We fill with 0 assuming 0 is the baseline/silence.
        df = df.fillna(0)

        # 1. Beamforming
        virtual_sensor = self._calculate_beamformed_sensor(df)
        df["virtual_source"] = virtual_sensor

        all_features = {}

        # 2. Extract features for Physical Sensors
        for col in SENSOR_COLS:
            sensor_feats = self._extract_series_features(df[col], col)
            all_features.update(sensor_feats)

        # 3. Extract features for Virtual Sensor
        virtual_feats = self._extract_series_features(
            df["virtual_source"], "virtual_source"
        )
        all_features.update(virtual_feats)

        # 4. Extract Spatial Features (Physical only)
        spatial_feats = self._extract_spatial_features(df)
        all_features.update(spatial_feats)

        # Add segment_id for joining later
        all_features["segment_id"] = int(segment_id)

        return all_features

    def generate_features(self, metadata_df, data_type="train", load_cached_data=True):
        """
        Main method to generate features for a dataset defined by metadata.

        Args:
            metadata_df: DataFrame containing 'segment_id' and 'file_path'.
            data_type: String identifier ('train', 'val', 'test') for caching.
            load_cached_data: Boolean, whether to attempt loading from cache.

        Returns:
            DataFrame containing extracted features.
        """
        # Define cache parameters
        # We include the length of metadata to invalidate cache if dataset size changes (e.g. debug mode)
        cache_params = {
            "data_type": data_type,
            "num_samples": len(metadata_df),
            "n_mfcc": MFCC_N_MFCC,
            "feature_version": "v1_beamforming",
        }
        cache_base_name = f"{data_type}_features"

        # 1. Try Load from Cache
        if load_cached_data:
            cached_df = self.cache_manager.load(
                cache_base_name, params=cache_params, ext=".parquet"
            )
            if cached_df is not None:
                print(f"Loaded {data_type} features from cache.")
                return cached_df

        # 2. Compute from Scratch
        print(f"Generating {data_type} features for {len(metadata_df)} segments...")

        feature_list = []

        # Iterate over metadata
        # Note: Not using tqdm to avoid progress bars in output as requested
        for _, row in metadata_df.iterrows():
            segment_id = row["segment_id"]
            file_path = row["file_path"]

            feats = self.process_segment(segment_id, file_path)
            if feats is not None:
                feature_list.append(feats)

        # Create DataFrame
        features_df = pd.DataFrame(feature_list)

        # 3. Save to Cache
        self.cache_manager.save(
            features_df, cache_base_name, params=cache_params, ext=".parquet"
        )
        print(f"Saved {data_type} features to cache.")

        return features_df


def get_tabular_features(train_meta, val_meta, test_meta, load_cached_data=True):
    """
    Wrapper function to generate features for all splits.
    """
    extractor = TabularFeatureExtractor()

    print("Processing Train Set...")
    train_features = extractor.generate_features(train_meta, "train", load_cached_data)

    print("Processing Validation Set...")
    val_features = extractor.generate_features(val_meta, "val", load_cached_data)

    print("Processing Test Set...")
    test_features = extractor.generate_features(test_meta, "test", load_cached_data)

    return train_features, val_features, test_features
