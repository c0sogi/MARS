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

        # 2. Windowing & Hierarchical Feature Extraction
        window_size = FEATURE_CONFIG["window_size"]
        overlap = FEATURE_CONFIG["window_overlap"]
        step = window_size - overlap

        # Prepare lists to store window-level stats
        # Structure: window_stats[sensor][stat_name] = [val_window_1, val_window_2, ...]
        window_data = {s: {} for s in sensors}

        num_rows = len(df)

        # Iterate over windows
        for start in range(0, num_rows - window_size + 1, step):
            end = start + window_size
            df_window = df.iloc[start:end]

            for sensor in sensors:
                signal = df_window[sensor].values

                # --- Time Domain Stats ---
                if "mean" in FEATURE_CONFIG["time_stats"]:
                    window_data[sensor].setdefault("mean", []).append(np.mean(signal))
                if "std" in FEATURE_CONFIG["time_stats"]:
                    window_data[sensor].setdefault("std", []).append(np.std(signal))
                if "min" in FEATURE_CONFIG["time_stats"]:
                    window_data[sensor].setdefault("min", []).append(np.min(signal))
                if "max" in FEATURE_CONFIG["time_stats"]:
                    window_data[sensor].setdefault("max", []).append(np.max(signal))
                if "skew" in FEATURE_CONFIG["time_stats"]:
                    window_data[sensor].setdefault("skew", []).append(
                        stats.skew(signal)
                    )
                if "kurtosis" in FEATURE_CONFIG["time_stats"]:
                    window_data[sensor].setdefault("kurtosis", []).append(
                        stats.kurtosis(signal)
                    )

                # Quantiles
                # Map config keys like 'q05' to percentile 5
                for stat in FEATURE_CONFIG["time_stats"]:
                    if stat.startswith("q") and stat[1:].isdigit():
                        q_val = int(stat[1:])
                        val = np.percentile(signal, q_val)
                        window_data[sensor].setdefault(stat, []).append(val)

                # --- Frequency Domain Stats ---
                # Only compute if needed
                if FEATURE_CONFIG["freq_stats"]:
                    # Compute FFT
                    fft_vals = np.fft.rfft(signal)
                    fft_power = np.abs(fft_vals) ** 2
                    freqs = np.fft.rfftfreq(
                        len(signal), d=1 / FEATURE_CONFIG["sampling_rate"]
                    )

                    # Spectral Centroid
                    if "spectral_centroid" in FEATURE_CONFIG["freq_stats"]:
                        sum_power = np.sum(fft_power)
                        if sum_power > 0:
                            centroid = np.sum(freqs * fft_power) / sum_power
                        else:
                            centroid = 0
                        window_data[sensor].setdefault("spectral_centroid", []).append(
                            centroid
                        )

                    # Dominant Frequency
                    if "dominant_freq" in FEATURE_CONFIG["freq_stats"]:
                        dom_freq = freqs[np.argmax(fft_power)]
                        window_data[sensor].setdefault("dominant_freq", []).append(
                            dom_freq
                        )

                    # Spectral Power Mean/Std
                    if "spectral_power_mean" in FEATURE_CONFIG["freq_stats"]:
                        window_data[sensor].setdefault(
                            "spectral_power_mean", []
                        ).append(np.mean(fft_power))
                    if "spectral_power_std" in FEATURE_CONFIG["freq_stats"]:
                        window_data[sensor].setdefault("spectral_power_std", []).append(
                            np.std(fft_power)
                        )

                    # Spectral Entropy
                    if "spectral_entropy" in FEATURE_CONFIG["freq_stats"]:
                        # Normalize power to treat as probability distribution
                        sum_power = np.sum(fft_power)
                        if sum_power > 0:
                            psd_norm = fft_power / sum_power
                            # Add epsilon to avoid log(0)
                            entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
                        else:
                            entropy = 0
                        window_data[sensor].setdefault("spectral_entropy", []).append(
                            entropy
                        )

        # 3. Aggregation (Level 2)
        # Collapse the lists of window stats into single values
        agg_funcs = FEATURE_CONFIG["aggregation_stats"]

        for sensor in sensors:
            for stat_name, values in window_data[sensor].items():
                values_arr = np.array(values)

                if len(values_arr) == 0:
                    # Handle case where no windows were processed (e.g. file too short)
                    # Fill with 0 or NaN
                    for agg in agg_funcs:
                        features[f"{sensor}_{stat_name}_{agg}"] = 0.0
                    continue

                if "mean" in agg_funcs:
                    features[f"{sensor}_{stat_name}_mean"] = np.mean(values_arr)
                if "std" in agg_funcs:
                    features[f"{sensor}_{stat_name}_std"] = np.std(values_arr)
                if "min" in agg_funcs:
                    features[f"{sensor}_{stat_name}_min"] = np.min(values_arr)
                if "max" in agg_funcs:
                    features[f"{sensor}_{stat_name}_max"] = np.max(values_arr)
                if "skew" in agg_funcs:
                    features[f"{sensor}_{stat_name}_skew"] = stats.skew(values_arr)

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
