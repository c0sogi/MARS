import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from library.config import FEATURE_CONFIG, INPUT_DIR, get_cache_path, N_JOBS
from library.utils import seed_everything

# Ensure reproducibility
seed_everything()


def extract_segment_features(segment_id, file_rel_path):
    """
    Extracts hierarchical time-frequency features for a single segment.

    Args:
        segment_id (int): The ID of the segment.
        file_rel_path (str): Relative path to the CSV file (e.g., 'train/123.csv').

    Returns:
        dict: A dictionary containing the extracted features and the segment_id.
    """
    file_path = os.path.join(INPUT_DIR, file_rel_path)

    try:
        # Load data
        # Using float32 to save memory, though calculations might need higher precision
        df = pd.read_csv(file_path, dtype="float32")

        # Preprocessing: Fill NaNs
        if FEATURE_CONFIG["fill_na_strategy"] == "mean":
            df = df.fillna(df.mean())
        else:
            df = df.fillna(0)

        # Handle case where fillna(mean) leaves NaNs (if column is all NaN)
        df = df.fillna(0)

        features = {"segment_id": segment_id}
        sensors = FEATURE_CONFIG["sensors"]

        # 1. Spatial Features (Correlation)
        if FEATURE_CONFIG["use_spatial_correlation"]:
            corr_matrix = df[sensors].corr()
            # Extract upper triangle to avoid duplicates and self-correlation
            for i in range(len(sensors)):
                for j in range(i + 1, len(sensors)):
                    s1, s2 = sensors[i], sensors[j]
                    features[f"corr_{s1}_{s2}"] = corr_matrix.loc[s1, s2]

        # 2. Global Feature Extraction (Cite solution_lesson_node_00004)
        # Compute statistics on the full segment directly
        for sensor in sensors:
            signal = df[sensor].values

            # --- Time Domain Stats ---
            if "mean" in FEATURE_CONFIG["stats"]:
                features[f"{sensor}_mean"] = np.mean(signal)
            if "std" in FEATURE_CONFIG["stats"]:
                features[f"{sensor}_std"] = np.std(signal)
            if "min" in FEATURE_CONFIG["stats"]:
                features[f"{sensor}_min"] = np.min(signal)
            if "max" in FEATURE_CONFIG["stats"]:
                features[f"{sensor}_max"] = np.max(signal)
            if "skew" in FEATURE_CONFIG["stats"]:
                features[f"{sensor}_skew"] = stats.skew(signal)
            if "kurtosis" in FEATURE_CONFIG["stats"]:
                features[f"{sensor}_kurtosis"] = stats.kurtosis(signal)
            if "mad" in FEATURE_CONFIG["stats"]:
                # Mean Absolute Deviation
                features[f"{sensor}_mad"] = np.mean(np.abs(signal - np.mean(signal)))

            # Quantiles
            for stat in FEATURE_CONFIG["stats"]:
                if stat.startswith("q") and stat[1:].isdigit():
                    q_val = int(stat[1:])
                    features[f"{sensor}_{stat}"] = np.percentile(signal, q_val)

            # --- Frequency Domain Stats ---
            if FEATURE_CONFIG["freq_stats"]:
                # Compute FFT once per sensor
                fft_vals = np.fft.rfft(signal)
                fft_power = np.abs(fft_vals) ** 2
                freqs = np.fft.rfftfreq(
                    len(signal), d=1 / FEATURE_CONFIG["sampling_rate"]
                )

                if "spectral_centroid" in FEATURE_CONFIG["freq_stats"]:
                    sum_power = np.sum(fft_power)
                    if sum_power > 0:
                        centroid = np.sum(freqs * fft_power) / sum_power
                    else:
                        centroid = 0
                    features[f"{sensor}_spectral_centroid"] = centroid

                if "dominant_freq" in FEATURE_CONFIG["freq_stats"]:
                    dom_freq = freqs[np.argmax(fft_power)]
                    features[f"{sensor}_dominant_freq"] = dom_freq

                if "spectral_power_mean" in FEATURE_CONFIG["freq_stats"]:
                    features[f"{sensor}_spectral_power_mean"] = np.mean(fft_power)

                if "spectral_power_std" in FEATURE_CONFIG["freq_stats"]:
                    features[f"{sensor}_spectral_power_std"] = np.std(fft_power)

                if "spectral_entropy" in FEATURE_CONFIG["freq_stats"]:
                    sum_power = np.sum(fft_power)
                    if sum_power > 0:
                        psd_norm = fft_power / sum_power
                        entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
                    else:
                        entropy = 0
                    features[f"{sensor}_spectral_entropy"] = entropy

        # --- Sanitize Features ---
        # Replace NaN and Infinity with 0.0 to prevent downstream errors
        for key, value in features.items():
            if isinstance(value, (int, float, np.number)):
                if np.isnan(value) or np.isinf(value):
                    features[key] = 0.0

        return features

    except Exception as e:
        print(f"Error processing {file_rel_path}: {e}")
        return None


class FeatureManager:
    """
    Manages the extraction, caching, and loading of features for train, val, and test sets.
    """

    def __init__(self):
        pass

    def _process_subset(self, metadata_df, subset_name, load_cached_data=True):
        """
        Generic method to load or compute features for a given subset.
        """
        cache_path = get_cache_path(subset_name)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {subset_name} features from {cache_path}...")
            try:
                df_features = pd.read_parquet(cache_path)
                # Ensure alignment with metadata
                # Merge on segment_id to ensure order matches metadata
                df_merged = metadata_df.merge(df_features, on="segment_id", how="left")

                # Check for missing features after merge
                if df_merged.isnull().any().any():
                    print(
                        f"Warning: Cached features for {subset_name} contain nulls or missing segments."
                    )
                    # Fill NaNs in loaded cache as well, just in case
                    df_merged = df_merged.fillna(0)

                # Separate features and target
                X = df_merged.drop(
                    columns=["segment_id", "time_to_eruption", "file_path"]
                )
                y = df_merged["time_to_eruption"]
                return X, y
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing features...")

        # 2. Compute features
        print(f"Computing features for {subset_name} ({len(metadata_df)} samples)...")

        # Use joblib for parallel processing
        results = Parallel(n_jobs=N_JOBS, verbose=0)(
            delayed(extract_segment_features)(row["segment_id"], row["file_path"])
            for _, row in metadata_df.iterrows()
        )

        # Filter out failed results (None)
        results = [r for r in results if r is not None]

        if not results:
            raise RuntimeError(
                f"Feature extraction failed for all files in {subset_name}."
            )

        df_features = pd.DataFrame(results)

        # Save to cache
        print(f"Saving {subset_name} features to {cache_path}...")
        df_features.to_parquet(cache_path, index=False)

        # Merge with metadata to ensure correct order and get targets
        df_merged = metadata_df.merge(df_features, on="segment_id", how="left")

        # Final safety fillna
        df_merged = df_merged.fillna(0)

        X = df_merged.drop(columns=["segment_id", "time_to_eruption", "file_path"])
        y = df_merged["time_to_eruption"]

        return X, y

    def get_train_data(self, metadata_df, load_cached_data=True):
        return self._process_subset(metadata_df, "train", load_cached_data)

    def get_val_data(self, metadata_df, load_cached_data=True):
        return self._process_subset(metadata_df, "val", load_cached_data)

    def get_test_data(self, metadata_df, load_cached_data=True):
        # For test, y will be all zeros, but we return it for consistency
        return self._process_subset(metadata_df, "test", load_cached_data)
