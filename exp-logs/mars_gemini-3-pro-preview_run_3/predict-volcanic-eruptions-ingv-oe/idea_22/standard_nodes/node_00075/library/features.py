import os
import numpy as np
import pandas as pd
import scipy.signal as signal
from joblib import Parallel, delayed
from scipy.stats import entropy, skew, kurtosis

from library.config import Config
from library.utils import setup_logger, reduce_mem_usage


class FeatureExtractor:
    """
    Implements the Hybrid-Transform Orthogonal Decomposition feature engineering strategy.
    Extracts Kinematic, Texture, Intensity, Spectral, and Temporal features.
    """

    def __init__(self):
        self.logger = setup_logger("FeatureExtractor")
        self.sensors = [f"sensor_{i}" for i in range(1, Config.N_SENSORS + 1)]

        # Configuration shortcuts
        self.sg_window = Config.SG_WINDOW
        self.sg_poly = Config.SG_POLYORDER
        self.quantiles = Config.QUANTILES
        self.wavelet = Config.WAVELET_TYPE
        self.welch_nperseg = Config.WELCH_NPERSEG
        self.temp_windows = Config.TEMPORAL_WINDOWS

    def _get_kinematics(self, x: np.ndarray, sensor_name: str) -> dict:
        """
        View 1: Kinematic Trend via Savitzky-Golay filters.
        Computes Velocity (1st deriv) and Acceleration (2nd deriv).
        """
        feats = {}

        # 1st Derivative (Velocity)
        vel = signal.savgol_filter(
            x, window_length=self.sg_window, polyorder=self.sg_poly, deriv=1
        )
        # 2nd Derivative (Acceleration)
        acc = signal.savgol_filter(
            x, window_length=self.sg_window, polyorder=self.sg_poly, deriv=2
        )

        # Dense Quantiles for Velocity
        vel_qs = np.quantile(vel, self.quantiles)
        for q, val in zip(self.quantiles, vel_qs):
            feats[f"{sensor_name}_vel_q{int(q*100)}"] = val

        # Shape Stats for Velocity (Cite Lesson 00021)
        feats[f"{sensor_name}_vel_skew"] = skew(vel)
        feats[f"{sensor_name}_vel_kurt"] = kurtosis(vel)

        # Dense Quantiles for Acceleration
        acc_qs = np.quantile(acc, self.quantiles)
        for q, val in zip(self.quantiles, acc_qs):
            feats[f"{sensor_name}_acc_q{int(q*100)}"] = val

        # Shape Stats for Acceleration (Cite Lesson 00021)
        feats[f"{sensor_name}_acc_skew"] = skew(acc)
        feats[f"{sensor_name}_acc_kurt"] = kurtosis(acc)

        return feats

    def _get_texture(self, x: np.ndarray, sensor_name: str) -> dict:
        """
        View 2: Texture via Residuals (Raw - Trend).
        Cite solution_lesson_node_00049.
        """
        feats = {}
        # Re-calculate Trend (fast enough to repeat)
        trend = signal.savgol_filter(
            x, window_length=self.sg_window, polyorder=self.sg_poly
        )
        # Residual (Texture)
        residual = x - trend

        # Texture Statistics
        feats[f"{sensor_name}_txt_rms"] = np.sqrt(np.mean(residual**2))
        feats[f"{sensor_name}_txt_skew"] = skew(residual)
        feats[f"{sensor_name}_txt_kurt"] = kurtosis(residual)

        return feats

    def _get_intensity(self, x: np.ndarray, sensor_name: str) -> dict:
        """
        View 3: Absolute Intensity via Raw Data stats.
        """
        feats = {}
        v_min = np.min(x)
        v_max = np.max(x)
        feats[f"{sensor_name}_raw_min"] = v_min
        feats[f"{sensor_name}_raw_max"] = v_max
        feats[f"{sensor_name}_raw_range"] = v_max - v_min
        return feats

    def _get_spectral(self, x: np.ndarray, sensor_name: str) -> dict:
        """
        View 4: Structural Spectral Features via PSD (Welch).
        """
        feats = {}

        # Compute PSD
        freqs, psd = signal.welch(x, nperseg=self.welch_nperseg)

        # Integrate power over 3 bands (Low, Mid, High)
        # Assuming freqs are 0 to 0.5 (normalized Nyquist) or similar.
        # We split the available frequency range into 3 equal parts.
        n_freqs = len(freqs)
        band_size = n_freqs // 3

        # Total Power
        total_power = np.sum(psd)
        feats[f"{sensor_name}_spec_total"] = total_power

        # Low Band
        low = np.sum(psd[:band_size])
        feats[f"{sensor_name}_spec_low"] = low

        # Mid Band
        mid = np.sum(psd[band_size : 2 * band_size])
        feats[f"{sensor_name}_spec_mid"] = mid

        # High Band
        high = np.sum(psd[2 * band_size :])
        feats[f"{sensor_name}_spec_high"] = high

        # Relative Power (Cite failure analysis from previous run)
        if total_power > 0:
            feats[f"{sensor_name}_spec_low_rel"] = low / total_power
            feats[f"{sensor_name}_spec_mid_rel"] = mid / total_power
            feats[f"{sensor_name}_spec_high_rel"] = high / total_power
        else:
            feats[f"{sensor_name}_spec_low_rel"] = 0
            feats[f"{sensor_name}_spec_mid_rel"] = 0
            feats[f"{sensor_name}_spec_high_rel"] = 0

        return feats

    def _get_temporal(self, x: np.ndarray, sensor_name: str) -> dict:
        """
        View 5: Temporal Evolution via Aggregated Window Statistics.
        Cite solution_lesson_node_00050.
        """
        feats = {}

        # Split into non-overlapping windows
        chunks = np.array_split(x, self.temp_windows)

        means = []
        rmses = []

        for chunk in chunks:
            if len(chunk) == 0:
                means.append(0)
                rmses.append(0)
            else:
                means.append(np.mean(chunk))
                rmses.append(np.sqrt(np.mean(chunk**2)))

        # Shift-Invariant Aggregates
        feats[f"{sensor_name}_win_mean_avg"] = np.mean(means)
        feats[f"{sensor_name}_win_mean_std"] = np.std(means)
        feats[f"{sensor_name}_win_mean_range"] = np.max(means) - np.min(means)

        feats[f"{sensor_name}_win_rms_avg"] = np.mean(rmses)
        feats[f"{sensor_name}_win_rms_std"] = np.std(rmses)
        feats[f"{sensor_name}_win_rms_range"] = np.max(rmses) - np.min(rmses)

        return feats

    def _process_single_file(self, row: pd.Series) -> dict:
        """
        Worker function to process a single CSV file.
        """
        segment_id = int(row["segment_id"])
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            # Load data
            df = pd.read_csv(file_path, dtype="float32")

            # Imputation: Fill NaNs with column mean (Segment-wise)
            df = df.fillna(df.mean())

            # If still NaNs (e.g., all NaN column), fill with 0
            df = df.fillna(0)

            feature_row = {"segment_id": segment_id}

            # Iterate over sensors
            for sensor in self.sensors:
                if sensor not in df.columns:
                    # Should not happen based on dataset description, but safety first
                    x = np.zeros(60001, dtype=np.float32)
                else:
                    x = df[sensor].values

                # Extract Views
                feature_row.update(self._get_kinematics(x, sensor))
                feature_row.update(self._get_texture(x, sensor))
                feature_row.update(self._get_intensity(x, sensor))
                feature_row.update(self._get_spectral(x, sensor))
                feature_row.update(self._get_temporal(x, sensor))

            return feature_row

        except Exception as e:
            self.logger.error(f"Error processing segment {segment_id}: {e}")
            return None

    def process_data(
        self, meta_df: pd.DataFrame, dataset_name: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Main method to process a dataset (train/val/test).
        Handles caching and parallel execution.

        Args:
            meta_df: Metadata DataFrame containing segment_ids and file_paths.
            dataset_name: Name of the dataset (e.g., 'train', 'val', 'test') for cache naming.
            load_cached_data: Whether to attempt loading from cache.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        cache_path = os.path.join(
            Config.WORKING_DIR, f"{dataset_name}_features.parquet"
        )

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(f"Loading cached features from {cache_path}")
            try:
                features_df = pd.read_parquet(cache_path)
                # Verify segment_ids match metadata
                if set(features_df["segment_id"]) == set(meta_df["segment_id"]):
                    return features_df
                else:
                    self.logger.warning(
                        "Cached data segment_ids do not match metadata. Recomputing..."
                    )
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute Features
        self.logger.info(
            f"Starting feature extraction for {dataset_name} ({len(meta_df)} files)..."
        )

        # Debugging: subset if configured
        if Config.DEBUG:
            self.logger.info(
                f"Debug mode: processing first {Config.DEBUG_SAMPLE_SIZE} files only."
            )
            meta_df = meta_df.head(Config.DEBUG_SAMPLE_SIZE)

        # Parallel Execution
        # n_jobs=-1 uses all available cores.
        # backend="loky" is robust for pandas/numpy operations.
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(self._process_single_file)(row) for _, row in meta_df.iterrows()
        )

        # Filter out failed items (None)
        results = [r for r in results if r is not None]

        if not results:
            raise ValueError("No features were extracted. Check input data paths.")

        features_df = pd.DataFrame(results)

        # Merge target if available in metadata
        if "time_to_eruption" in meta_df.columns:
            features_df = features_df.merge(
                meta_df[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
            )

        # Optimize Memory
        features_df = reduce_mem_usage(features_df, verbose=False)

        # 3. Save Cache
        self.logger.info(f"Saving features to {cache_path}")
        features_df.to_parquet(cache_path, index=False)

        return features_df
