import os
import numpy as np
import pandas as pd
from scipy import signal, stats

try:
    import pywt
except ImportError:
    pywt = None
from joblib import Parallel, delayed
from library import config

# ==========================================
# Constants & Configuration
# ==========================================
FS = 100.0  # Sampling frequency assumed to be 100Hz (60000 samples / 600 seconds)

# ==========================================
# Helper Functions
# ==========================================


def calculate_moments(x, prefix):
    """
    Calculates Mean, Std, Skewness, and Kurtosis for a given signal.
    """
    if len(x) == 0:
        return {
            f"{prefix}_mean": 0,
            f"{prefix}_std": 0,
            f"{prefix}_skew": 0,
            f"{prefix}_kurt": 0,
        }

    mean_x = np.mean(x)
    std_x = np.std(x)

    # Handle zero variance (Cite debug_lesson_4)
    if std_x == 0 or np.isnan(std_x):
        skew_x = 0
        kurt_x = 0
    else:
        skew_x = stats.skew(x)
        kurt_x = stats.kurtosis(x)

    return {
        f"{prefix}_mean": mean_x,
        f"{prefix}_std": std_x,
        f"{prefix}_skew": skew_x,
        f"{prefix}_kurt": kurt_x,
    }


def calculate_spectral_features(x, fs=FS, nperseg=config.WELCH_NPERSEG):
    """
    Computes PSD Band Power using Welch's Method.
    Bands: Low (0.1-3Hz), Mid (3-10Hz), High (10-45Hz).
    """
    try:
        freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg)
    except Exception:
        # Fallback if signal is too short for nperseg
        freqs, psd = signal.welch(x, fs=fs, nperseg=min(len(x), 256))

    # Define bands
    bands = {"band_low": (0.1, 3.0), "band_mid": (3.0, 10.0), "band_high": (10.0, 45.0)}

    features = {}

    # Frequency resolution
    delta_f = freqs[1] - freqs[0]

    for band_name, (low_f, high_f) in bands.items():
        # Find indices
        idx = np.logical_and(freqs >= low_f, freqs <= high_f)
        # Integrate PSD (sum * delta_f)
        band_power = np.sum(psd[idx]) * delta_f
        features[band_name] = band_power

    return features


def calculate_wavelet_features(x, wavelet=config.WAVELET_TYPE):
    """
    Applies DWT and extracts Energy and Entropy from detail coefficients.
    """
    try:
        # Single level decomposition
        cA, cD = pywt.dwt(x, wavelet)

        # Use detail coefficients (Texture)
        energy = np.sum(cD**2) / len(cD)

        # Shannon Entropy of the squared coefficients (normalized)
        # Treat squared coefficients as a probability distribution
        p = cD**2
        p_sum = np.sum(p)
        if p_sum > 0:
            p_norm = p / p_sum
            # Avoid log(0)
            p_norm = p_norm[p_norm > 0]
            entropy = -np.sum(p_norm * np.log(p_norm))
        else:
            entropy = 0.0

        return {"wavelet_energy": energy, "wavelet_entropy": entropy}
    except Exception:
        return {"wavelet_energy": 0.0, "wavelet_entropy": 0.0}


def calculate_temporal_profile(x, n_segments=config.N_TEMPORAL_SEGMENTS):
    """
    Splits signal into n_segments and computes RMS and Mean for each.
    """
    features = {}

    # Calculate segment length
    L = len(x)
    seg_len = L // n_segments

    for i in range(n_segments):
        start = i * seg_len
        # For the last segment, include any remainder
        end = (i + 1) * seg_len if i < n_segments - 1 else L

        segment = x[start:end]

        if len(segment) > 0:
            seg_mean = np.mean(segment)
            seg_rms = np.sqrt(np.mean(segment**2))
        else:
            seg_mean = 0.0
            seg_rms = 0.0

        features[f"win_{i}_mean"] = seg_mean
        features[f"win_{i}_rms"] = seg_rms

    return features


def extract_segment_features(file_path, segment_id):
    """
    Core function to process a single sensor file.
    Implements the High-Resolution Orthogonal Decomposition.
    """
    try:
        # Load Data
        df = pd.read_csv(file_path, dtype="float32")

        # Imputation: Fill NaNs with column means
        # Cite debug_lesson_1: Chain fillna(0) to handle columns that are all NaN
        df = df.fillna(df.mean()).fillna(0)

        features = {"segment_id": int(segment_id)}

        for sensor in config.SENSOR_COLS:
            if sensor not in df.columns:
                continue

            raw_signal = df[sensor].values

            # ==========================================
            # Decomposition
            # ==========================================

            # View A: Trend (Savitzky-Golay)
            # Use polyorder 2, window size 51 (from config)
            try:
                trend = signal.savgol_filter(
                    raw_signal,
                    window_length=config.SG_WINDOW_SIZE,
                    polyorder=config.SG_POLYORDER,
                )
            except Exception:
                # Fallback if signal is shorter than window
                trend = raw_signal

            # View B: Texture (Residuals)
            texture = raw_signal - trend

            # View C: Raw (Energy)
            # (Already in raw_signal)

            # ==========================================
            # Feature Extraction
            # ==========================================

            prefix = sensor

            # 1. View A (Kinematics)
            # ----------------------
            # 0th derivative (Position/Trend)
            features.update(calculate_moments(trend, f"{prefix}_trend"))

            # 1st derivative (Velocity)
            velocity = np.diff(trend, prepend=trend[0])
            features.update(calculate_moments(velocity, f"{prefix}_vel"))

            # 2nd derivative (Acceleration)
            acceleration = np.diff(velocity, prepend=velocity[0])
            features.update(calculate_moments(acceleration, f"{prefix}_acc"))

            # 2. View B (Wavelet Texture)
            # ---------------------------
            # DWT on Residuals
            features.update(
                {
                    f"{prefix}_{k}": v
                    for k, v in calculate_wavelet_features(texture).items()
                }
            )

            # 3. View C (Physical Intensity)
            # ------------------------------
            features[f"{prefix}_min"] = np.min(raw_signal)
            features[f"{prefix}_max"] = np.max(raw_signal)
            features[f"{prefix}_ptp"] = np.ptp(raw_signal)

            # 4. View C (High-Res Spectral)
            # -----------------------------
            # Welch PSD on Raw Signal
            features.update(
                {
                    f"{prefix}_{k}": v
                    for k, v in calculate_spectral_features(raw_signal).items()
                }
            )

            # 5. View C (Flattened Temporal Profiling)
            # ----------------------------------------
            # Windowed statistics on Raw Signal
            features.update(
                {
                    f"{prefix}_{k}": v
                    for k, v in calculate_temporal_profile(raw_signal).items()
                }
            )

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ==========================================
# Main Processing Logic
# ==========================================


def process_metadata_split(metadata_path, split_name, load_cached_data=True):
    """
    Processes a specific metadata split (train, val, or test).
    Handles caching via Parquet.
    """
    # Define cache path
    cache_file = os.path.join(config.WORKING_DIR, f"{split_name}_features.parquet")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached features for {split_name} from {cache_file}...")
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)
    print(f"Processing {len(meta_df)} files for {split_name}...")

    # Prepare arguments for parallel execution
    # Construct full paths
    tasks = []
    for _, row in meta_df.iterrows():
        file_path = os.path.join(config.INPUT_DIR, row["file_path"])
        segment_id = row["segment_id"]
        tasks.append((file_path, segment_id))

    # Parallel Execution
    # n_jobs=-1 uses all available cores
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(extract_segment_features)(fp, sid) for fp, sid in tasks
    )

    # Filter out failed results (None)
    results = [r for r in results if r is not None]

    # Create DataFrame
    features_df = pd.DataFrame(results)

    # Merge target if available (train/val)
    if "time_to_eruption" in meta_df.columns:
        features_df = features_df.merge(
            meta_df[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
        )

    # 3. Save to cache
    print(f"Saving {split_name} features to {cache_file}...")
    features_df.to_parquet(cache_file, index=False)

    return features_df


def get_train_data(load_cached_data=True):
    return process_metadata_split(config.TRAIN_METADATA_PATH, "train", load_cached_data)


def get_val_data(load_cached_data=True):
    return process_metadata_split(config.VAL_METADATA_PATH, "val", load_cached_data)


def get_test_data(load_cached_data=True):
    return process_metadata_split(config.TEST_METADATA_PATH, "test", load_cached_data)
