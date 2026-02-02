import os
import numpy as np
import pandas as pd
import librosa
import joblib
from joblib import Parallel, delayed
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("feature_engineering")


def extract_robust_tabular_features(df_segment):
    """
    Extracts robust audio-seismic features from a multi-sensor dataframe.

    Args:
        df_segment (pd.DataFrame): Dataframe with shape (n_samples, n_sensors).
                                   Missing values should be handled before calling this.

    Returns:
        dict: Dictionary containing flattened features.
    """
    features = {}
    sensor_cols = [c for c in df_segment.columns if "sensor" in c]

    # 1. Per-Sensor Features
    for sensor in sensor_cols:
        sig = df_segment[sensor].values.astype(np.float32)

        # --- Time Domain Statistics ---
        # Robust statistics (Mean, Std, Q05, Q95) - No Min/Max
        mean_val = np.mean(sig)
        std_val = np.std(sig)
        q05 = np.quantile(sig, 0.05)
        q95 = np.quantile(sig, 0.95)

        features[f"{sensor}_mean"] = mean_val
        features[f"{sensor}_std"] = std_val
        features[f"{sensor}_q05"] = q05
        features[f"{sensor}_q95"] = q95

        # Absolute Quantiles (Energy/Magnitude structure)
        abs_sig = np.abs(sig)
        features[f"{sensor}_abs_q05"] = np.quantile(abs_sig, 0.05)
        features[f"{sensor}_abs_q95"] = np.quantile(abs_sig, 0.95)

        # Raw Zero-Crossing Rate (without centering)
        # Using simple sign change count
        zcr = ((sig[:-1] * sig[1:]) < 0).sum() / len(sig)
        features[f"{sensor}_zcr"] = zcr

        # --- Frequency Domain (MFCCs) ---
        # Limit to first 13 coefficients (Timbre/Envelope)
        # n_mfcc=13 returns 13 coefficients per frame
        mfcc = librosa.feature.mfcc(
            y=sig,
            sr=Config.SAMPLING_RATE,
            n_mfcc=Config.N_MFCC,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
        )
        # mfcc shape: (n_mfcc, t)

        # Aggregate MFCCs over time using robust stats
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_q05 = np.quantile(mfcc, 0.05, axis=1)
        mfcc_q95 = np.quantile(mfcc, 0.95, axis=1)

        for i in range(Config.N_MFCC):
            features[f"{sensor}_mfcc_{i}_mean"] = mfcc_mean[i]
            features[f"{sensor}_mfcc_{i}_std"] = mfcc_std[i]
            features[f"{sensor}_mfcc_{i}_q05"] = mfcc_q05[i]
            features[f"{sensor}_mfcc_{i}_q95"] = mfcc_q95[i]

    # 2. Spatial Interaction (Correlation Matrix)
    # Compute correlation between all pairs
    corr_matrix = df_segment[sensor_cols].corr().values

    # Extract upper triangle indices (excluding diagonal)
    # 10 sensors -> 45 pairs
    triu_indices = np.triu_indices(len(sensor_cols), k=1)
    corr_values = corr_matrix[triu_indices]

    k = 0
    for i in range(len(sensor_cols)):
        for j in range(i + 1, len(sensor_cols)):
            features[f"corr_{sensor_cols[i]}_{sensor_cols[j]}"] = corr_values[k]
            k += 1

    return features


def generate_log_mel_spectrogram(df_segment):
    """
    Generates a 10-channel Log-Mel Spectrogram tensor.

    Args:
        df_segment (pd.DataFrame): Dataframe with shape (n_samples, n_sensors).

    Returns:
        np.ndarray: Shape (n_sensors, n_mels, time_steps)
    """
    sensor_cols = [c for c in df_segment.columns if "sensor" in c]
    specs = []

    for sensor in sensor_cols:
        sig = df_segment[sensor].values.astype(np.float32)

        # Compute Mel Spectrogram
        melspec = librosa.feature.melspectrogram(
            y=sig,
            sr=Config.SAMPLING_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            fmin=Config.FMIN,
            fmax=Config.FMAX,
        )

        # Convert to Log Scale (dB)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)

        # Normalize to 0-1 range or strictly standardize?
        # The Config says "Instance Standardization" happens later (likely in dataset),
        # but here we just return the raw log-mel features.
        # However, to ensure numerical stability, we keep it as dB.

        specs.append(log_melspec)

    # Stack to create (10, n_mels, time)
    # Expected shape: (10, 128, ~118)
    return np.stack(specs, axis=0)


def _process_single_file(row):
    """
    Helper function to process a single file for parallel execution.
    """
    segment_id = row["segment_id"]
    file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

    try:
        # Load data
        # Use float32 to save memory and handle potential NaNs
        df = pd.read_csv(file_path, dtype="float32")

        # Handle Missing Values (Mean Imputation per segment)
        # This is crucial for FFT/Spectral analysis to avoid artifacts
        df = df.fillna(df.mean())

        # Fallback for columns that might be all NaN
        df = df.fillna(0)

        # Extract Features
        tabular_feats = extract_robust_tabular_features(df)
        tabular_feats["segment_id"] = segment_id

        spectrogram = generate_log_mel_spectrogram(df)

        return tabular_feats, spectrogram

    except Exception as e:
        logger.error(f"Error processing segment {segment_id}: {e}")
        return None, None


def process_data(metadata_path, dataset_name, load_cached_data=True):
    """
    Main driver function to process a dataset (train/val/test).
    Handles caching, parallel processing, and data aggregation.

    Args:
        metadata_path (str): Path to the metadata CSV.
        dataset_name (str): Name tag for the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (df_tabular, X_spectrograms, y_targets)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    tab_cache_path = os.path.join(cache_dir, f"{dataset_name}_tabular.parquet")
    spec_cache_path = os.path.join(cache_dir, f"{dataset_name}_spectrograms.npy")
    target_cache_path = os.path.join(cache_dir, f"{dataset_name}_targets.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(tab_cache_path)
            and os.path.exists(spec_cache_path)
            and os.path.exists(target_cache_path)
        ):

            logger.info(f"Loading cached data for {dataset_name}...")
            df_tabular = pd.read_parquet(tab_cache_path)
            X_spectrograms = np.load(spec_cache_path)
            y_targets = np.load(target_cache_path)

            logger.info(
                f"Loaded {dataset_name}: Tabular {df_tabular.shape}, Spec {X_spectrograms.shape}"
            )
            return df_tabular, X_spectrograms, y_targets
        else:
            logger.info(f"Cache missing for {dataset_name}, processing from scratch...")

    # 2. Process from Scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # For debugging, we might want to limit size, but here we process full
    # If DEBUG is set in Config, we could slice df_meta, but usually
    # feature engineering should run on full data to be ready.
    if Config.DEBUG:
        logger.info("DEBUG mode: processing subset of data")
        df_meta = df_meta.head(50)

    logger.info(f"Processing {len(df_meta)} files for {dataset_name}...")

    # Run parallel processing
    results = Parallel(n_jobs=Config.NUM_WORKERS, backend="loky")(
        delayed(_process_single_file)(row) for _, row in df_meta.iterrows()
    )

    # Filter out failures
    results = [r for r in results if r[0] is not None]

    if not results:
        raise RuntimeError("No data processed successfully.")

    # Unzip results
    tabular_list, spec_list = zip(*results)

    # Aggregate Tabular
    df_tabular = pd.DataFrame(tabular_list)

    # Merge with target from metadata
    # Ensure order is preserved or merge on segment_id
    df_tabular = df_tabular.merge(
        df_meta[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
    )

    # Aggregate Spectrograms
    X_spectrograms = np.stack(spec_list, axis=0)

    # Extract Targets
    y_targets = df_tabular["time_to_eruption"].values

    # Drop target from tabular features to keep X clean (optional, but good practice to separate)
    # However, for saving, keeping it in dataframe is convenient.
    # We will return it separately as requested.

    # 3. Save to Cache
    logger.info(f"Saving cache for {dataset_name}...")
    df_tabular.to_parquet(tab_cache_path, index=False)
    np.save(spec_cache_path, X_spectrograms)
    np.save(target_cache_path, y_targets)

    logger.info(
        f"Processed {dataset_name}: Tabular {df_tabular.shape}, Spec {X_spectrograms.shape}"
    )

    return df_tabular, X_spectrograms, y_targets
