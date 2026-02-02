import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from library.config import Config


class TabularFeatureExtractor:
    """
    Extracts expert features from sensor data for Branch A (LightGBM).
    Handles data loading, feature engineering, and caching.
    """

    def __init__(self):
        self.input_dir = Config.INPUT_DIR
        self.working_dir = Config.WORKING_DIR
        self.sensors = Config.SENSORS

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def _calculate_haar_features(self, x):
        """
        Calculates simple Haar Wavelet features (Energy, Entropy) manually
        to avoid dependency on PyWavelets.

        Args:
            x (np.array): Input signal.

        Returns:
            tuple: (energy, entropy)
        """
        # Ensure even length for simple splitting
        if len(x) % 2 != 0:
            x = x[:-1]

        # Vectorized Haar decomposition
        # Detail coefficients: (x[2n] - x[2n+1]) / sqrt(2)
        # We just want the high-frequency component (Detail)
        # x[::2] are even indices, x[1::2] are odd indices
        detail = (x[::2] - x[1::2]) / np.sqrt(2)

        # Energy
        energy = np.sum(detail**2)

        # Entropy (Shannon entropy of normalized energy distribution)
        # Add epsilon to avoid log(0)
        p = (detail**2) / (energy + 1e-10)
        entropy = -np.sum(p * np.log(p + 1e-10))

        return energy, entropy

    def _process_segment(self, segment_id, file_path, target=None):
        """
        Reads a single segment CSV and computes features.

        Args:
            segment_id (int): ID of the segment.
            file_path (str): Relative path to the CSV file.
            target (float, optional): Target value.

        Returns:
            dict: Dictionary of extracted features.
        """
        full_path = os.path.join(self.input_dir, file_path)

        features = {"segment_id": segment_id}
        if target is not None:
            features["time_to_eruption"] = target

        try:
            # Load data
            # Use float32 to save memory, handle NaNs
            if not os.path.exists(full_path):
                # Should not happen based on metadata verification, but safe fallback
                return features

            df = pd.read_csv(full_path, dtype="float32")

            # Fill NaNs with 0 (common in seismic data if sensor fails)
            df = df.fillna(0)

            for sensor in self.sensors:
                if sensor not in df.columns:
                    continue

                x = df[sensor].values

                # --- Time Domain Statistics ---
                features[f"{sensor}_mean"] = np.mean(x)
                features[f"{sensor}_std"] = np.std(x)
                features[f"{sensor}_min"] = np.min(x)
                features[f"{sensor}_max"] = np.max(x)
                features[f"{sensor}_skew"] = stats.skew(x)
                features[f"{sensor}_kurt"] = stats.kurtosis(x)

                # Quantiles
                q = np.quantile(x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
                features[f"{sensor}_q01"] = q[0]
                features[f"{sensor}_q05"] = q[1]
                features[f"{sensor}_q25"] = q[2]
                features[f"{sensor}_q50"] = q[3]
                features[f"{sensor}_q75"] = q[4]
                features[f"{sensor}_q95"] = q[5]
                features[f"{sensor}_q99"] = q[6]

                # Absolute Statistics
                abs_x = np.abs(x)
                features[f"{sensor}_abs_mean"] = np.mean(abs_x)
                features[f"{sensor}_abs_max"] = np.max(abs_x)

                # Structural / Temporal
                # Total Variation (sum of absolute differences)
                diff_x = np.diff(x)
                features[f"{sensor}_total_variation"] = np.sum(np.abs(diff_x))
                features[f"{sensor}_mean_change_abs"] = np.mean(np.abs(diff_x))

                # Zero Crossing Rate
                zcr = ((x[:-1] * x[1:]) < 0).sum()
                features[f"{sensor}_zcr"] = zcr

                # --- Frequency Domain (FFT) ---
                # Real FFT
                fft_x = np.fft.rfft(x)
                fft_mag = np.abs(fft_x)

                # Normalize magnitude by length
                fft_mag = fft_mag / len(x)

                features[f"{sensor}_fft_mean"] = np.mean(fft_mag)
                features[f"{sensor}_fft_std"] = np.std(fft_mag)
                features[f"{sensor}_fft_max"] = np.max(fft_mag)

                # Dominant Frequency (index of max)
                features[f"{sensor}_fft_dom_freq_idx"] = np.argmax(fft_mag)

                # Spectral Skew/Kurtosis
                features[f"{sensor}_fft_skew"] = stats.skew(fft_mag)
                features[f"{sensor}_fft_kurt"] = stats.kurtosis(fft_mag)

                # --- Wavelet Domain (Haar) ---
                haar_energy, haar_entropy = self._calculate_haar_features(x)
                features[f"{sensor}_haar_energy"] = haar_energy
                features[f"{sensor}_haar_entropy"] = haar_entropy

        except Exception as e:
            # In case of corrupt file, we return what we have (likely just ID/target)
            print(f"Error processing {file_path}: {e}")
            pass

        return features

    def get_features(self, dataset_type="train", load_cached_data=True):
        """
        Main method to get features for a dataset.

        Args:
            dataset_type (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series or None): Target variable.
            ids (pd.Series): Segment IDs.
        """
        cache_file = f"tabular_features_{dataset_type}.parquet"
        cache_path = os.path.join(self.working_dir, cache_file)

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(
                f"Loading cached tabular features for {dataset_type} from {cache_path}"
            )
            df = pd.read_parquet(cache_path)
        else:
            # 2. Compute Features
            print(f"Computing tabular features for {dataset_type}...")

            # Identify Metadata File
            if dataset_type == "train":
                meta_path = Config.TRAIN_METADATA
            elif dataset_type == "val":
                meta_path = Config.VAL_METADATA
            elif dataset_type == "test":
                meta_path = Config.TEST_METADATA
            else:
                raise ValueError(f"Invalid dataset_type: {dataset_type}")

            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            df_meta = pd.read_csv(meta_path)

            # Prepare arguments for parallel processing
            tasks = []
            for _, row in df_meta.iterrows():
                seg_id = row["segment_id"]
                f_path = row["file_path"]
                # Target is 0 for test, actual value for train/val
                target = row["time_to_eruption"] if "time_to_eruption" in row else None
                tasks.append((seg_id, f_path, target))

            # Execute Parallel Processing
            # Use n_jobs=-1 to use all CPUs
            results = Parallel(n_jobs=-1, verbose=0)(
                delayed(self._process_segment)(sid, fp, tgt) for sid, fp, tgt in tasks
            )

            # Create DataFrame
            df = pd.DataFrame(results)

            # Save to Cache
            print(f"Saving tabular features to {cache_path}")
            df.to_parquet(cache_path, index=False)

        # 3. Separate X, y, ids
        ids = df["segment_id"]

        if dataset_type == "test":
            y = None
            # Drop target if it exists (it's just 0s)
            if "time_to_eruption" in df.columns:
                X = df.drop(columns=["time_to_eruption", "segment_id"])
            else:
                X = df.drop(columns=["segment_id"])
        else:
            # For train/val, we expect targets
            if "time_to_eruption" in df.columns:
                y = df["time_to_eruption"]
                X = df.drop(columns=["time_to_eruption", "segment_id"])
            else:
                # Should not happen for train/val unless cache is corrupted
                y = None
                X = df.drop(columns=["segment_id"])

        return X, y, ids
