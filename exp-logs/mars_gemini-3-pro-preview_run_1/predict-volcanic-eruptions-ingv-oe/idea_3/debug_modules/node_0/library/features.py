import os
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed
import warnings

from library.config import Config
from library.utils import get_config_hash

# Suppress warnings
warnings.filterwarnings("ignore")


def calculate_entropy(x):
    """
    Calculate Shannon Entropy of a signal's energy distribution.
    """
    # Energy of the signal
    energy = x**2
    total_energy = np.sum(energy)

    if total_energy == 0:
        return 0.0

    # Probability distribution of energy
    p = energy / total_energy

    # Filter out zero probabilities to avoid log(0)
    p = p[p > 0]

    # Shannon Entropy: -sum(p * log(p))
    return -np.sum(p * np.log(p))


def get_global_stats(series, prefix):
    """
    Compute global statistical features: Mean, Std, Skew, Kurtosis, Min, Max, Quantiles, TV.
    """
    x = series.values
    stats = {}

    # Basic Stats
    stats[f"{prefix}_mean"] = np.mean(x)
    stats[f"{prefix}_std"] = np.std(x)
    stats[f"{prefix}_min"] = np.min(x)
    stats[f"{prefix}_max"] = np.max(x)

    # Higher Order Moments
    stats[f"{prefix}_skew"] = skew(x)
    stats[f"{prefix}_kurt"] = kurtosis(x)

    # Quantiles (1%, 5%, 95%, 99%)
    # We use numpy quantile for efficiency
    q = np.quantile(x, [0.01, 0.05, 0.95, 0.99])
    stats[f"{prefix}_q01"] = q[0]
    stats[f"{prefix}_q05"] = q[1]
    stats[f"{prefix}_q95"] = q[2]
    stats[f"{prefix}_q99"] = q[3]

    # Structural: Total Variation (sum of absolute differences)
    stats[f"{prefix}_tv"] = np.sum(np.abs(np.diff(x)))

    return stats


def get_zero_crossing_rate(series, prefix):
    """
    Compute Raw Zero-Crossing Rate.
    """
    x = series.values
    # Count sign changes
    zcr = ((x[:-1] * x[1:]) < 0).sum()
    return {f"{prefix}_zcr": zcr}


def get_fft_features(series, prefix):
    """
    Compute FFT-based spectral features.
    """
    x = series.values

    # Compute FFT
    fft_vals = fft(x)

    # Magnitude Spectrum
    fft_abs = np.abs(fft_vals)

    # Take first half (symmetric for real inputs)
    n = len(x)
    fft_abs = fft_abs[: n // 2]

    stats = {}
    stats[f"{prefix}_fft_mean"] = np.mean(fft_abs)
    stats[f"{prefix}_fft_std"] = np.std(fft_abs)
    stats[f"{prefix}_fft_max"] = np.max(fft_abs)  # Magnitude of dominant frequency
    stats[f"{prefix}_fft_skew"] = skew(fft_abs)
    stats[f"{prefix}_fft_kurt"] = kurtosis(fft_abs)

    return stats


def get_wavelet_features(series, prefix, wavelet="db4"):
    """
    Compute Discrete Wavelet Transform features manually (since pywt might not be available).
    Implements a single-level decomposition using Daubechies-4 coefficients.
    """
    x = series.values

    # Daubechies 4 coefficients
    s3 = np.sqrt(3)
    c0 = (1 + s3) / (4 * np.sqrt(2))
    c1 = (3 + s3) / (4 * np.sqrt(2))
    c2 = (3 - s3) / (4 * np.sqrt(2))
    c3 = (1 - s3) / (4 * np.sqrt(2))

    # Low-pass filter (Approximation)
    # h = [c0, c1, c2, c3]
    # High-pass filter (Detail)
    # g = [c3, -c2, c1, -c0]

    # Convolve and downsample (step=2)
    # mode='valid' avoids boundary padding artifacts
    cA = np.convolve(x, [c0, c1, c2, c3], mode="valid")[::2]
    cD = np.convolve(x, [c3, -c2, c1, -c0], mode="valid")[::2]

    stats = {}
    # Energy
    stats[f"{prefix}_wd_encA"] = np.sum(cA**2)
    stats[f"{prefix}_wd_encD"] = np.sum(cD**2)

    # Entropy
    stats[f"{prefix}_wd_entA"] = calculate_entropy(cA)
    stats[f"{prefix}_wd_entD"] = calculate_entropy(cD)

    return stats


def extract_segment_features(segment_id, file_path, sensors, wavelet):
    """
    Extracts all features for a single data segment.
    """
    # Read CSV
    # Use float32 to handle NaNs and reduce memory usage
    try:
        df = pd.read_csv(file_path, dtype="float32")
    except FileNotFoundError:
        return None

    # Fill NaNs:
    # 1. Fill with column mean (best guess)
    # 2. Fill remaining (if column is all NaN) with 0
    df = df.fillna(df.mean()).fillna(0)

    features = {"segment_id": int(segment_id)}

    for sensor in sensors:
        # Check if sensor exists in file
        if sensor not in df.columns:
            continue

        series = df[sensor]
        prefix = sensor

        # 1. Global Stats
        features.update(get_global_stats(series, prefix))

        # 2. Zero Crossing Rate
        features.update(get_zero_crossing_rate(series, prefix))

        # 3. FFT Features
        features.update(get_fft_features(series, prefix))

        # 4. Wavelet Features
        features.update(get_wavelet_features(series, prefix, wavelet))

    return features


def generate_feature_matrix(
    metadata_path, output_path=None, load_cached_data=True, split_name="train"
):
    """
    Main driver function to generate or load the feature matrix for a dataset split.

    Args:
        metadata_path (str): Path to the metadata CSV (train.csv, val.csv, or test.csv).
        output_path (str): Optional path to save the output (legacy support).
                           The function primarily uses hashed filenames in the working dir.
        load_cached_data (bool): If True, attempts to load from cache first.
        split_name (str): Name of the split ('train', 'val', 'test') for file naming.

    Returns:
        pd.DataFrame: The processed feature matrix.
    """
    # 1. Define Feature Configuration for Hashing
    feature_config = {
        "sensors": Config.SENSORS,
        "wavelet": Config.WAVELET_FAMILY,
        "quantiles": [0.01, 0.05, 0.95, 0.99],
        "features": ["global", "zcr", "fft", "wavelet_db4_manual"],
        "version": "1.0",
    }
    config_hash = get_config_hash(feature_config)

    # 2. Determine Cache Path
    # We use the hash to ensure that if logic changes, we recompute.
    cache_filename = f"{split_name}_features_{config_hash}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 3. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features for {split_name} from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Generating features for {split_name} (Hash: {config_hash})...")

    # 4. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # 5. Prepare for Parallel Processing
    # Metadata file_path is relative to input dir, e.g., "train/123.csv"
    full_paths = [os.path.join(Config.INPUT_DIR, p) for p in df_meta["file_path"]]
    segment_ids = df_meta["segment_id"].values

    # 6. Execute Parallel Feature Extraction
    # n_jobs=-1 uses all available CPU cores
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(extract_segment_features)(
            seg_id, path, Config.SENSORS, Config.WAVELET_FAMILY
        )
        for seg_id, path in zip(segment_ids, full_paths)
    )

    # Filter out any failed reads (None)
    results = [r for r in results if r is not None]

    # Convert to DataFrame
    df_features = pd.DataFrame(results)

    # 7. Merge Targets (if available in metadata)
    if "time_to_eruption" in df_meta.columns:
        # Ensure segment_id types match
        df_meta["segment_id"] = df_meta["segment_id"].astype(int)
        df_features["segment_id"] = df_features["segment_id"].astype(int)

        # Merge target column
        df_features = df_features.merge(
            df_meta[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
        )

    # 8. Save to Cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df_features.to_parquet(cache_path, index=False)
    print(f"Saved features to {cache_path}")

    # Also save to the output_path if provided (for compatibility with fixed paths in Config)
    if output_path:
        # Create directory for output path if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_features.to_parquet(output_path, index=False)
        print(f"Saved copy to {output_path}")

    return df_features
