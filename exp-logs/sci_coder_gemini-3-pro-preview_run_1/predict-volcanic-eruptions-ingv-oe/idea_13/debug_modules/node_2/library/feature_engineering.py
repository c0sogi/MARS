import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torchaudio
from joblib import Parallel, delayed
from library.config import Config
from library.utils import save_npy, load_npy, save_parquet, load_parquet

# ==========================================
# Core Signal Processing Functions
# ==========================================


def extract_tabular_features(df_segment):
    """
    Extracts energy-partitioned and robust statistical features from a sensor segment.
    """
    features = {}

    # Fill NaNs with column means to handle missing sensor data
    df_segment = df_segment.fillna(df_segment.mean()).fillna(0)

    # Pre-compute numpy array for speed
    signals = df_segment.values.T  # Shape: (n_sensors, n_timesteps)

    # 1. Spatial Correlation (Sensor Interactions)
    # Correlation matrix of sensors
    corr_matrix = np.corrcoef(signals)
    # Extract upper triangle indices (excluding diagonal)
    upper_indices = np.triu_indices(Config.NUM_SENSORS, k=1)
    corr_values = corr_matrix[upper_indices]

    for i, val in enumerate(corr_values):
        features[f"spatial_corr_{i}"] = val

    # Define MFCC transform
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=Config.SAMPLING_RATE,
        n_mfcc=Config.N_MFCC,
        melkwargs={
            "n_fft": Config.N_FFT,
            "hop_length": Config.HOP_LENGTH,
            "n_mels": Config.N_MELS,
            "center": True,
        },
    )

    # Iterate over each sensor
    for sensor_idx, sensor_name in enumerate(Config.SENSOR_COLS):
        sig = signals[sensor_idx]

        # --- Time Domain Stats ---
        # Absolute Quantiles and Extremes
        features[f"{sensor_name}_min"] = np.min(sig)
        features[f"{sensor_name}_max"] = np.max(sig)
        features[f"{sensor_name}_mean"] = np.mean(sig)
        features[f"{sensor_name}_std"] = np.std(sig)
        features[f"{sensor_name}_kurtosis"] = stats.kurtosis(sig)

        # Crest Factor (Impulsiveness): Max Abs / RMS
        rms = np.sqrt(np.mean(sig**2))
        max_abs = np.max(np.abs(sig))
        crest_factor = max_abs / (rms + 1e-9)
        features[f"{sensor_name}_crest_factor"] = crest_factor

        # --- Frequency Domain: Sub-band Energies ---
        # Compute FFT
        fft_vals = np.fft.rfft(sig)
        fft_freqs = np.fft.rfftfreq(len(sig), d=1 / Config.SAMPLING_RATE)
        power_spectrum = np.abs(fft_vals) ** 2

        for low_f, high_f in Config.SUBBANDS:
            # Mask for specific band
            mask = (fft_freqs >= low_f) & (fft_freqs < high_f)
            if np.sum(mask) > 0:
                band_energy = np.sum(power_spectrum[mask])
                # Log-Subband Energy
                features[f"{sensor_name}_band_{low_f}_{high_f}_energy"] = np.log1p(
                    band_energy
                )
            else:
                features[f"{sensor_name}_band_{low_f}_{high_f}_energy"] = 0.0

        # --- Cepstral Domain: Robust MFCCs ---
        # We use a smaller n_fft for MFCC to capture texture, or standard.
        # Using Config.N_FFT is fine.
        # Torchaudio expects tensor input
        sig_tensor = torch.from_numpy(sig).float()
        mfccs = mfcc_transform(sig_tensor).numpy()

        # Skip 0th coefficient (energy) as we have explicit energy features
        mfccs = mfccs[1:, :]

        # Robust Aggregation (Median, Q05, Q95)
        features[f"{sensor_name}_mfcc_median"] = np.median(mfccs)
        features[f"{sensor_name}_mfcc_q05"] = np.percentile(mfccs, 5)
        features[f"{sensor_name}_mfcc_q95"] = np.percentile(mfccs, 95)
        features[f"{sensor_name}_mfcc_iqr"] = stats.iqr(mfccs)

    return features


def extract_injection_scalars(df_segment):
    """
    Extracts global energy statistics for the Vision Model's scalar injection layer.
    Returns a numpy array of shape (SCALAR_INPUT_DIM,).
    """
    df_segment = df_segment.fillna(df_segment.mean()).fillna(0)
    signals = df_segment.values.T

    scalar_features = []

    for sig in signals:
        # 1. Log Total Energy
        energy = np.sum(sig**2)
        log_energy = np.log1p(energy)

        # 2. Global Max (Abs)
        g_max = np.max(np.abs(sig))

        # 3. Crest Factor
        rms = np.sqrt(np.mean(sig**2))
        crest = g_max / (rms + 1e-9)

        scalar_features.extend([log_energy, g_max, crest])

    return np.array(scalar_features, dtype=np.float32)


def generate_spectrogram(df_segment, global_max_val):
    """
    Generates Log-Mel Spectrograms for all sensors and normalizes them
    using the global maximum value.
    Returns array of shape (n_sensors, n_mels, n_timesteps).
    """
    df_segment = df_segment.fillna(df_segment.mean()).fillna(0)
    signals = df_segment.values.T

    specs = []

    # Define MelSpectrogram transform
    melspec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLING_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
        center=True,
        power=2.0,
    )

    for sig in signals:
        # Compute Mel Spectrogram
        sig_tensor = torch.from_numpy(sig).float()
        melspec = melspec_transform(sig_tensor).numpy()

        # Log scaling
        log_melspec = np.log1p(melspec)

        # Global Max Normalization
        # Denominator is log(global_max + 1)
        norm_factor = np.log1p(global_max_val)
        if norm_factor > 1e-9:
            norm_melspec = log_melspec / norm_factor
        else:
            norm_melspec = log_melspec

        specs.append(norm_melspec)

    # Stack to (C, H, W)
    specs = np.stack(specs, axis=0)

    # Ensure fixed width if necessary (resize or crop)
    # Current settings: 60001 / 256 ~ 235 frames.
    # Config.IMG_SIZE is (128, 256). We pad or crop width to 256.
    target_width = Config.IMG_SIZE[1]
    current_width = specs.shape[2]

    if current_width < target_width:
        pad_width = target_width - current_width
        specs = np.pad(specs, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
    elif current_width > target_width:
        specs = specs[:, :, :target_width]

    return specs.astype(np.float32)


# ==========================================
# Data Processing Pipeline
# ==========================================


def get_global_max(load_cached=True):
    """
    Computes or loads the global maximum absolute value across the training set.
    Used for fixed-scale normalization.
    """
    if load_cached and os.path.exists(Config.CACHE_GLOBAL_MAX):
        return float(load_npy(Config.CACHE_GLOBAL_MAX))

    print("Computing global maximum from training data...")
    df_train = pd.read_csv(Config.TRAIN_METADATA)

    # Helper to get max of a single file
    def get_file_max(row):
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        if not os.path.exists(path):
            return 0.0
        try:
            df = pd.read_csv(path, dtype="float32")
            return df.abs().max().max()
        except:
            return 0.0

    # Run in parallel
    max_values = Parallel(n_jobs=Config.NUM_WORKERS)(
        delayed(get_file_max)(row) for _, row in df_train.iterrows()
    )

    global_max = max(max_values) if max_values else 1.0
    save_npy(np.array(global_max), Config.CACHE_GLOBAL_MAX)
    print(f"Global max computed: {global_max}")
    return float(global_max)


def _process_single_file(row, global_max, output_spec_dir):
    """
    Worker function to process a single data file.
    Returns: (segment_id, tabular_features_dict, scalar_features_array)
    """
    segment_id = row["segment_id"]
    file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

    if not os.path.exists(file_path):
        # Return None or empty if file missing (should not happen based on metadata check)
        return None

    try:
        df_seg = pd.read_csv(file_path, dtype="float32")

        # 1. Tabular Features
        tab_feats = extract_tabular_features(df_seg)
        tab_feats["segment_id"] = segment_id

        # 2. Injection Scalars
        scalars = extract_injection_scalars(df_seg)

        # 3. Spectrogram (Save to disk immediately)
        spec = generate_spectrogram(df_seg, global_max)
        spec_save_path = os.path.join(output_spec_dir, f"{segment_id}.npy")
        save_npy(spec, spec_save_path)

        return (tab_feats, scalars)

    except Exception as e:
        print(f"Error processing {segment_id}: {e}")
        return None


def process_dataset(
    metadata_path, cache_tabular_path, cache_spec_dir, global_max, load_cached=True
):
    """
    Generic function to process a dataset (Train, Val, or Test).
    """
    if load_cached and os.path.exists(cache_tabular_path):
        # Check if spectrogram dir is populated
        if len(os.listdir(cache_spec_dir)) > 0:
            print(f"Loading cached tabular features from {cache_tabular_path}")
            df_tabular = load_parquet(cache_tabular_path)

            # We also need to return scalars.
            # For simplicity in this pipeline, we will attach scalars to the tabular dataframe
            # with specific column names so they can be separated later,
            # OR we assume the tabular cache is sufficient and scalars are re-computed
            # or stored in the parquet.
            # Strategy: Store scalars in the parquet with prefix "scalar_".
            return df_tabular

    print(f"Processing data from {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)

    if Config.DEBUG:
        df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)

    results = Parallel(n_jobs=Config.NUM_WORKERS)(
        delayed(_process_single_file)(row, global_max, cache_spec_dir)
        for _, row in df_meta.iterrows()
    )

    # Filter Nones
    results = [r for r in results if r is not None]

    if not results:
        return pd.DataFrame()

    # Unpack results
    tabular_list = []
    for tab_feats, scalars in results:
        # Flatten scalars into the dict for storage
        for i, val in enumerate(scalars):
            tab_feats[f"scalar_{i}"] = val
        tabular_list.append(tab_feats)

    df_tabular = pd.DataFrame(tabular_list)

    # Merge with targets if available
    if "time_to_eruption" in df_meta.columns:
        # Ensure alignment
        df_tabular = df_tabular.merge(
            df_meta[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
        )

    save_parquet(df_tabular, cache_tabular_path)
    print(f"Saved tabular features to {cache_tabular_path}")

    return df_tabular


def generate_features(load_cached=True):
    """
    Main entry point to generate all features for Train, Val, and Test.
    """
    # 1. Get Global Max (needed for spectrogram normalization)
    global_max = get_global_max(load_cached=load_cached)

    # 2. Process Train
    df_train = process_dataset(
        Config.TRAIN_METADATA,
        Config.CACHE_TRAIN_FEATURES,
        Config.CACHE_SPECTROGRAMS_TRAIN,
        global_max,
        load_cached,
    )

    # 3. Process Val
    df_val = process_dataset(
        Config.VAL_METADATA,
        Config.CACHE_VAL_FEATURES,
        Config.CACHE_SPECTROGRAMS_VAL,
        global_max,
        load_cached,
    )

    # 4. Process Test
    df_test = process_dataset(
        Config.TEST_METADATA,
        Config.CACHE_TEST_FEATURES,
        Config.CACHE_SPECTROGRAMS_TEST,
        global_max,
        load_cached,
    )

    return df_train, df_val, df_test
