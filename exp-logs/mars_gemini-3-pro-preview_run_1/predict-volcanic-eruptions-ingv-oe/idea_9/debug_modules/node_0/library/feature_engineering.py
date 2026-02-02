import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed
from library.config import Config


def compute_beamformed_sensor(df_segment: pd.DataFrame) -> pd.Series:
    """
    Computes a virtual beamformed sensor by averaging normalized signals.
    Implements Delay-and-Sum Beamforming assuming simultaneous arrival.
    """
    # Select physical sensors
    sensor_data = df_segment[Config.SENSOR_COLS].copy()

    # Fill NaNs with 0 (equilibrium) before normalization
    sensor_data = sensor_data.fillna(0)

    # Normalize each sensor: (x - mean) / (std + epsilon)
    # This ensures each sensor contributes equally to the sum
    means = sensor_data.mean()
    stds = sensor_data.std()
    normalized = (sensor_data - means) / (stds + 1e-6)

    # Compute mean across sensors (Beamforming)
    virtual_sensor = normalized.mean(axis=1)
    return virtual_sensor


def extract_features_for_segment(segment_df: pd.DataFrame, segment_id: int) -> dict:
    """
    Extracts parsimonious features for a single segment.
    """
    features = {}
    features["segment_id"] = segment_id

    # 1. Preprocessing
    # Fill missing values
    segment_df = segment_df.fillna(0)

    # Add Beamformed Virtual Sensor
    segment_df["sensor_beamformed"] = compute_beamformed_sensor(segment_df)

    # List of channels to process: 10 physical + 1 virtual
    all_channels = Config.SENSOR_COLS + ["sensor_beamformed"]

    # 2. Per-Sensor Feature Extraction
    for col in all_channels:
        x = segment_df[col].values

        # --- Global Statistics ---
        features[f"{col}_mean"] = np.mean(x)
        features[f"{col}_std"] = np.std(x)
        features[f"{col}_skew"] = skew(x)
        features[f"{col}_kurtosis"] = kurtosis(x)

        # Quantiles
        q_vals = np.quantile(x, [0.01, 0.05, 0.95, 0.99])
        features[f"{col}_q01"] = q_vals[0]
        features[f"{col}_q05"] = q_vals[1]
        features[f"{col}_q95"] = q_vals[2]
        features[f"{col}_q99"] = q_vals[3]

        # --- Absolute Statistics (Energy) ---
        abs_x = np.abs(x)
        features[f"{col}_abs_mean"] = np.mean(abs_x)
        features[f"{col}_abs_max"] = np.max(abs_x)

        abs_q_vals = np.quantile(abs_x, [0.05, 0.50, 0.95, 0.99])
        features[f"{col}_abs_q05"] = abs_q_vals[0]
        features[f"{col}_abs_median"] = abs_q_vals[1]
        features[f"{col}_abs_q95"] = abs_q_vals[2]
        features[f"{col}_abs_q99"] = abs_q_vals[3]

        # --- Structural Features ---
        # Raw Zero-Crossing Rate (no centering)
        # Using the signal offset as an amplitude gate
        zcr = ((x[:-1] * x[1:]) < 0).sum()
        features[f"{col}_zcr"] = zcr

        # --- Parsimonious MFCCs ---
        # Extract MFCCs (Coeffs 1-13)
        # We use Config params for consistency, though N_FFT/HOP might be tuned for spectrograms
        # For tabular, we just need consistent spectral features.
        # SR=100Hz.
        mfccs = librosa.feature.mfcc(
            y=x.astype(np.float32),
            sr=Config.SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
        )

        # Aggregate MFCCs using Robust Statistics ONLY
        # Exclude Min/Max to avoid outlier artifacts
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_q05 = np.quantile(mfccs, 0.05, axis=1)
        mfcc_q95 = np.quantile(mfccs, 0.95, axis=1)

        for i in range(Config.N_MFCC):
            features[f"{col}_mfcc_{i}_mean"] = mfcc_mean[i]
            features[f"{col}_mfcc_{i}_std"] = mfcc_std[i]
            features[f"{col}_mfcc_{i}_q05"] = mfcc_q05[i]
            features[f"{col}_mfcc_{i}_q95"] = mfcc_q95[i]

    # 3. Spatial Features (Correlation Matrix)
    # Compute correlation between physical sensors only
    corr_matrix = segment_df[Config.SENSOR_COLS].corr().values

    # Flatten upper triangle (excluding diagonal)
    k = 0
    for i in range(Config.NUM_SENSORS):
        for j in range(i + 1, Config.NUM_SENSORS):
            features[f"corr_s{i+1}_s{j+1}"] = corr_matrix[i, j]
            k += 1

    return features


def _process_single_file(row, input_dir):
    """
    Helper function for parallel processing.
    """
    segment_id = row["segment_id"]
    file_path = row["file_path"]
    full_path = os.path.join(input_dir, file_path)

    try:
        # Load data (float32 to save memory/speed)
        df = pd.read_csv(full_path, dtype="float32")

        # Extract features
        features = extract_features_for_segment(df, int(segment_id))
        return features
    except Exception as e:
        print(f"Error processing segment {segment_id}: {e}")
        return None


def process_dataset(dataset_type: str = "train", load_cached_data: bool = True):
    """
    Main function to process a dataset (train/val/test).
    Handles caching and parallel execution.

    Args:
        dataset_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (np.array): Target array (or None for test).
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    features_path = os.path.join(cache_dir, f"{dataset_type}_features.parquet")
    targets_path = os.path.join(cache_dir, f"{dataset_type}_targets.npy")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(features_path):
        print(f"Loading {dataset_type} features from cache: {features_path}")
        X = pd.read_parquet(features_path)

        y = None
        if os.path.exists(targets_path):
            y = np.load(targets_path)

        return X, y

    # 2. Process from Scratch
    print(f"Generating {dataset_type} features from scratch...")

    # Load Metadata
    if dataset_type == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif dataset_type == "val":
        meta_path = Config.VAL_METADATA_PATH
    elif dataset_type == "test":
        meta_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    df_meta = pd.read_csv(meta_path)

    # Debug Mode: Sample subset
    if Config.DEBUG:
        print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)

    # Parallel Feature Extraction
    # Use joblib to parallelize across files
    results = Parallel(n_jobs=Config.NUM_WORKERS, verbose=0)(
        delayed(_process_single_file)(row, Config.INPUT_DIR)
        for _, row in df_meta.iterrows()
    )

    # Filter out None results (errors)
    results = [r for r in results if r is not None]

    # Create DataFrame
    X = pd.DataFrame(results)

    # Handle Targets
    y = None
    if "time_to_eruption" in df_meta.columns and dataset_type != "test":
        # Align targets with successfully processed segments
        # We merge on segment_id to ensure alignment
        X_merged = X.merge(
            df_meta[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
        )
        y = X_merged["time_to_eruption"].values
        # Drop target and ID from features
        X = X_merged.drop(columns=["time_to_eruption"])

    # Ensure segment_id is not a feature for the model, but keep it for tracking if needed
    # Usually we drop it or set as index. Here we set as index.
    if "segment_id" in X.columns:
        X = X.set_index("segment_id")

    # 3. Save to Cache
    print(f"Saving {dataset_type} features to cache...")
    X.to_parquet(features_path)
    if y is not None:
        np.save(targets_path, y)

    return X, y
