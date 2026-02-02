import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torchaudio
import torchaudio.transforms as T
from library.config import Config
from library.utils import save_to_parquet, load_from_parquet, save_to_npy, load_from_npy


class DataManager:
    """
    Manages data loading, feature extraction, and caching for the volcano eruption prediction task.
    Handles tabular expert features (Branch A).
    """

    def __init__(self):
        self.tabular_config = Config.TABULAR_CONFIG
        self.sensors = Config.SENSORS

    def _extract_tabular_features(self, df_segment):
        """
        Computes statistical and spectral features for a single segment (all sensors).

        Args:
            df_segment (pd.DataFrame): 60001 rows x 10 columns (sensors).

        Returns:
            dict: Dictionary of extracted features.
        """
        features = {}

        # Pre-compute common aggregations to save time
        # We assume NaNs are already handled before this call or handled here
        # Filling NaNs with column mean for robustness
        df_segment = df_segment.fillna(df_segment.mean())

        for sensor in self.sensors:
            if sensor not in df_segment.columns:
                continue

            x = df_segment[sensor].values

            # --- Base Signal Statistics ---
            features[f"{sensor}_mean"] = np.mean(x)
            features[f"{sensor}_std"] = np.std(x)
            features[f"{sensor}_min"] = np.min(x)
            features[f"{sensor}_max"] = np.max(x)
            features[f"{sensor}_skew"] = float(stats.skew(x))
            features[f"{sensor}_kurtosis"] = float(stats.kurtosis(x))
            features[f"{sensor}_mad"] = np.mean(np.abs(x - np.mean(x)))

            # Shape Factors (Cite solution_lesson_node_00014: Magnitude-Aware)
            rms = np.sqrt(np.mean(x**2))
            mean_abs = np.mean(np.abs(x))
            if mean_abs > 0:
                features[f"{sensor}_shape_factor"] = rms / mean_abs
                features[f"{sensor}_impulse_factor"] = np.max(np.abs(x)) / mean_abs
            if rms > 0:
                features[f"{sensor}_crest_factor"] = np.max(np.abs(x)) / rms

            # Quantiles
            quantiles = np.quantile(x, self.tabular_config["quantiles"])
            for q, val in zip(self.tabular_config["quantiles"], quantiles):
                features[f"{sensor}_q{int(q*100)}"] = val

            # Absolute Quantiles (magnitude distribution)
            x_abs = np.abs(x)
            abs_quantiles = np.quantile(x_abs, self.tabular_config["abs_quantiles"])
            for q, val in zip(self.tabular_config["abs_quantiles"], abs_quantiles):
                features[f"{sensor}_abs_q{int(q*100)}"] = val

            # --- Gradient Statistics (Velocity/Acceleration) ---
            # Cite solution_lesson_node_00006: Structural metrics
            dx = np.diff(x)
            features[f"{sensor}_grad_mean"] = np.mean(dx)
            features[f"{sensor}_grad_std"] = np.std(dx)
            features[f"{sensor}_grad_max"] = np.max(dx)
            features[f"{sensor}_grad_min"] = np.min(dx)

            ddx = np.diff(dx)
            features[f"{sensor}_grad2_mean"] = np.mean(ddx)
            features[f"{sensor}_grad2_std"] = np.std(ddx)

            # Structural Features
            # Zero Crossing Rate: count sign changes
            zcr = ((x[:-1] * x[1:]) < 0).sum()
            features[f"{sensor}_zcr"] = zcr

            # Total Variation: sum of absolute differences
            tv = np.sum(np.abs(dx))
            features[f"{sensor}_tv"] = tv

            # --- Frequency Domain Statistics (FFT) ---
            # Real FFT
            fft_vals = np.fft.rfft(x)
            fft_mag = np.abs(fft_vals)

            features[f"{sensor}_fft_mean"] = np.mean(fft_mag)
            features[f"{sensor}_fft_std"] = np.std(fft_mag)
            features[f"{sensor}_fft_max"] = np.max(
                fft_mag
            )  # Dominant frequency magnitude

            # Spectral Centroid (approximate via weighted mean of indices)
            # Avoid division by zero
            sum_mag = np.sum(fft_mag)
            if sum_mag > 0:
                features[f"{sensor}_fft_centroid"] = (
                    np.sum(np.arange(len(fft_mag)) * fft_mag) / sum_mag
                )
            else:
                features[f"{sensor}_fft_centroid"] = 0.0

        return features

    def get_data(self, split, load_cached_data=True, debug=False):
        """
        Main method to retrieve data for a specific split (train/val/test).
        Handles caching logic.

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.
            debug (bool): If True, processes only a small subset.

        Returns:
            tuple: (X_tabular (pd.DataFrame), None, y (np.ndarray))
        """
        # Determine paths
        tabular_cache_path = Config.get_cache_path(split, "parquet")
        target_cache_path = os.path.join(Config.WORKING_DIR, f"{split}_targets.npy")

        # 1. Try Loading from Cache
        if load_cached_data and not debug:
            X_tab = load_from_parquet(tabular_cache_path)
            y = load_from_npy(target_cache_path)

            if X_tab is not None and y is not None:
                print(f"[{split.upper()}] Loaded data from cache.")
                return X_tab, None, y
            else:
                print(f"[{split.upper()}] Cache miss or incomplete. Reprocessing...")

        # 2. Load Metadata
        if split == "train":
            meta_path = Config.TRAIN_METADATA
        elif split == "val":
            meta_path = Config.VAL_METADATA
        elif split == "test":
            meta_path = Config.TEST_METADATA
        else:
            raise ValueError(f"Unknown split: {split}")

        df_meta = pd.read_csv(meta_path)

        if debug:
            df_meta = df_meta.head(20)
            print(f"[{split.upper()}] Debug mode: processing {len(df_meta)} samples.")

        # 3. Processing Loop
        tabular_list = []
        target_list = []

        print(f"[{split.upper()}] Processing {len(df_meta)} files...")

        for idx, row in df_meta.iterrows():
            segment_id = int(row["segment_id"])
            target = row["time_to_eruption"]

            # Construct file path (metadata contains relative path)
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            if not os.path.exists(file_path):
                # Should not happen based on metadata verification, but safe to skip or error
                print(f"Warning: File not found {file_path}")
                continue

            # Read CSV
            # Using float32 to save memory and match torch default
            try:
                df_seg = pd.read_csv(file_path, dtype="float32")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # A. Extract Tabular Features
            feats = self._extract_tabular_features(df_seg)
            feats["segment_id"] = segment_id  # Keep ID for reference if needed
            tabular_list.append(feats)

            # C. Store Target
            target_list.append(target)

            if (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1}/{len(df_meta)}...")

        # 4. Aggregate
        X_tab = pd.DataFrame(tabular_list)
        y = np.array(target_list, dtype=np.float32)

        # 5. Save to Cache (if not debugging)
        if not debug:
            print(f"[{split.upper()}] Saving to cache...")
            save_to_parquet(X_tab, tabular_cache_path)
            save_to_npy(y, target_cache_path)
            print(f"[{split.upper()}] Caching complete.")

        return X_tab, None, y
