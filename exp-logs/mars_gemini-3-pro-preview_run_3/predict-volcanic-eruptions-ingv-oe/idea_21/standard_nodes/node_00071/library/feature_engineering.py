import os
import pandas as pd
import numpy as np
import scipy.signal
import scipy.stats
from concurrent.futures import ProcessPoolExecutor
import library.config as config

# ==========================================
# Wavelet Coefficients (Daubechies 4)
# ==========================================
# Decomposition Low-pass Filter
DB4_LO_D = np.array(
    [
        -0.010597401785,
        0.032883011667,
        0.030841381836,
        -0.187034811719,
        -0.027983769417,
        0.630880767940,
        0.714846570553,
        0.230377813309,
    ]
)
# Decomposition High-pass Filter
DB4_HI_D = np.array(
    [
        -0.230377813309,
        0.714846570553,
        -0.630880767940,
        -0.027983769417,
        0.187034811719,
        0.030841381836,
        -0.032883011667,
        -0.010597401785,
    ]
)


def calculate_entropy(x):
    """Computes Shannon entropy of the distribution of x."""
    # Normalize to treat as probabilities
    x_abs = np.abs(x)
    total_energy = np.sum(x_abs)
    if total_energy == 0:
        return 0.0
    p = x_abs / total_energy
    # Filter zeros to avoid log(0)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def get_kinematic_features(series, window, polyorder, quantiles):
    """
    View 1: Kinematic Trend via Savitzky-Golay.
    Computes Velocity (1st deriv) and Acceleration (2nd deriv).
    """
    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    # Smooth and derivatives
    # deriv=0 is smoothed signal, 1 is velocity, 2 is acceleration
    try:
        velocity = scipy.signal.savgol_filter(
            series, window_length=window, polyorder=polyorder, deriv=1
        )
        acceleration = scipy.signal.savgol_filter(
            series, window_length=window, polyorder=polyorder, deriv=2
        )
    except ValueError:
        # Fallback for very short signals (unlikely given dataset)
        velocity = np.zeros_like(series)
        acceleration = np.zeros_like(series)

    feats = {}

    # Velocity Quantiles
    vel_quants = np.quantile(velocity, quantiles)
    for q, val in zip(quantiles, vel_quants):
        feats[f"vel_q{int(q*100)}"] = val

    # Acceleration Quantiles
    acc_quants = np.quantile(acceleration, quantiles)
    for q, val in zip(quantiles, acc_quants):
        feats[f"acc_q{int(q*100)}"] = val

    # Basic dispersion of kinematics
    feats["vel_std"] = np.std(velocity)
    feats["acc_std"] = np.std(acceleration)

    return feats


def get_texture_features(series, level):
    """
    View 2: Multi-Resolution Texture via Manual DWT (db4).
    Extracts Energy and Entropy from Detail Coefficients at each level.
    """
    feats = {}
    approx = series.copy()

    for l in range(1, level + 1):
        # Convolution mode 'valid' avoids boundary artifacts but shrinks signal
        # We need to handle edges, but for feature extraction on long signals 'valid' is acceptable
        # or 'same' with padding. Let's use 'full' and truncate to match DWT standard logic roughly.
        # Standard DWT: convolve then downsample.

        # High Pass -> Detail
        # Note: scipy.signal.convolve flips the kernel.
        # To match DWT filter definition, we usually don't flip if defined as filter coeffs,
        # but here we just want a consistent projection.
        d = scipy.signal.convolve(approx, DB4_HI_D, mode="valid")[::2]

        # Low Pass -> Approximation (for next level)
        a = scipy.signal.convolve(approx, DB4_LO_D, mode="valid")[::2]

        # Features from Detail Coefficients
        feats[f"dwt_l{l}_energy"] = np.sum(d**2) / (len(d) + 1e-9)  # Mean Energy
        feats[f"dwt_l{l}_entropy"] = calculate_entropy(d)
        feats[f"dwt_l{l}_max"] = np.max(np.abs(d))
        feats[f"dwt_l{l}_std"] = np.std(d)

        approx = a

    return feats


def get_intensity_features(series):
    """
    View 3: Absolute Intensity.
    """
    return {
        "min": np.min(series),
        "max": np.max(series),
        "range": np.max(series) - np.min(series),
        "mean_abs": np.mean(np.abs(series)),
    }


def get_spectral_features(series, fs, bands):
    """
    View 4: Structural Spectral Features via PSD (Welch).
    """
    # nperseg=256 is standard default, allows decent frequency resolution for 100Hz
    f, Pxx = scipy.signal.welch(series, fs=fs, nperseg=256)

    feats = {}
    total_power = np.sum(Pxx)
    if total_power == 0:
        total_power = 1e-9

    for band_name, (low_f, high_f) in bands.items():
        # Find indices
        idx = np.logical_and(f >= low_f, f <= high_f)
        band_power = np.sum(Pxx[idx])
        feats[f"psd_{band_name}_pwr"] = band_power
        feats[f"psd_{band_name}_rel"] = band_power / total_power

    return feats


def get_temporal_features(series, num_windows):
    """
    View 5: Temporal Evolution via Aggregation.
    Cite solution_lesson_node_00050: Shift Invariance vs Temporal Specificity.
    """
    # Split array into num_windows
    chunks = np.array_split(series, num_windows)

    win_rms = []

    for chunk in chunks:
        if len(chunk) > 0:
            rms = np.sqrt(np.mean(chunk**2))
            win_rms.append(rms)
        else:
            win_rms.append(0)

    win_rms = np.array(win_rms)

    # Aggregate window statistics (Shift Invariant)
    feats = {}
    feats["win_rms_mean"] = np.mean(win_rms)
    feats["win_rms_std"] = np.std(win_rms)
    feats["win_rms_min"] = np.min(win_rms)
    feats["win_rms_max"] = np.max(win_rms)
    feats["win_rms_skew"] = scipy.stats.skew(win_rms)

    return feats


def extract_features_for_segment(segment_id, file_path):
    """
    Master function to process one segment file.
    """
    full_path = os.path.join(config.INPUT_DIR, file_path)

    try:
        # Load data, float32 for memory efficiency
        df = pd.read_csv(full_path, dtype="float32")
    except FileNotFoundError:
        print(f"Warning: File {full_path} not found.")
        return None

    # Imputation: Fill NaNs with column mean
    df = df.fillna(df.mean()).fillna(0)

    all_features = {"segment_id": segment_id}

    # Iterate over sensors
    sensors = [f"sensor_{i}" for i in range(1, 11)]

    for sensor in sensors:
        if sensor not in df.columns:
            continue

        sig = df[sensor].values

        # Ensure odd window for SavGol
        window = config.SG_WINDOW if config.SG_WINDOW % 2 != 0 else config.SG_WINDOW + 1

        # 1. Decomposition (Cite solution_lesson_node_00049)
        # Trend via Savitzky-Golay
        try:
            trend = scipy.signal.savgol_filter(
                sig, window, config.SG_POLYORDER, deriv=0
            )
            # Texture = Residual
            texture = sig - trend
        except ValueError:
            trend = np.zeros_like(sig)
            texture = sig

        # 2. Kinematic Features (on Trend)
        # Velocity
        try:
            vel = scipy.signal.savgol_filter(sig, window, config.SG_POLYORDER, deriv=1)
            acc = scipy.signal.savgol_filter(sig, window, config.SG_POLYORDER, deriv=2)
        except ValueError:
            vel = np.zeros_like(sig)
            acc = np.zeros_like(sig)

        k_feats = {}
        k_feats.update(get_distribution_features(vel, "vel"))
        k_feats.update(get_distribution_features(acc, "acc"))

        for k, v in k_feats.items():
            all_features[f"{sensor}_{k}"] = v

        # 3. Texture Features (on Residual)
        t_feats = get_distribution_features(texture, "text")
        for k, v in t_feats.items():
            all_features[f"{sensor}_{k}"] = v

        # 4. Spectral Features (on Raw)
        s_feats = get_spectral_features(sig, config.SAMPLING_RATE, config.PSD_BANDS)
        for k, v in s_feats.items():
            all_features[f"{sensor}_{k}"] = v

        # 5. Temporal Features (on Raw, Aggregated)
        tm_feats = get_temporal_features(sig, config.TEMPORAL_NUM_WINDOWS)
        for k, v in tm_feats.items():
            all_features[f"{sensor}_{k}"] = v

    return all_features


def _process_wrapper(args):
    """Helper for parallel processing"""
    return extract_features_for_segment(*args)


def process_dataset(metadata_path, output_filename, load_cached_data=True):
    """
    Processes a dataset defined by metadata_path.
    Handles caching via parquet.
    """
    cache_path = os.path.join(config.WORKING_DIR, output_filename)

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing features for {metadata_path}...")

    # 2. Load Metadata
    meta_df = pd.read_csv(metadata_path)

    # Debugging: Sample subset if configured
    if config.DEBUG:
        print(f"DEBUG MODE: Sampling {config.DEBUG_SAMPLE_SIZE} rows.")
        meta_df = meta_df.head(config.DEBUG_SAMPLE_SIZE)

    # 3. Parallel Feature Extraction
    tasks = []
    for _, row in meta_df.iterrows():
        tasks.append((row["segment_id"], row["file_path"]))

    results = []
    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=config.N_JOBS) as executor:
        # map returns results in order
        for res in executor.map(_process_wrapper, tasks):
            if res is not None:
                results.append(res)

    # 4. Create DataFrame
    feature_df = pd.DataFrame(results)

    # Merge target if available (Train/Val)
    if "time_to_eruption" in meta_df.columns:
        feature_df = feature_df.merge(
            meta_df[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
        )

    # 5. Save Cache
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    feature_df.to_parquet(cache_path, index=False)
    print(f"Saved features to {cache_path}. Shape: {feature_df.shape}")

    return feature_df


def generate_train_val_test_features(load_cached_data=True):
    """
    Main entry point to generate all feature sets.
    """
    train_meta = os.path.join(config.METADATA_DIR, "train.csv")
    val_meta = os.path.join(config.METADATA_DIR, "val.csv")
    test_meta = os.path.join(config.METADATA_DIR, "test.csv")

    train_df = process_dataset(train_meta, "train_features.parquet", load_cached_data)
    val_df = process_dataset(val_meta, "val_features.parquet", load_cached_data)
    test_df = process_dataset(test_meta, "test_features.parquet", load_cached_data)

    return train_df, val_df, test_df
