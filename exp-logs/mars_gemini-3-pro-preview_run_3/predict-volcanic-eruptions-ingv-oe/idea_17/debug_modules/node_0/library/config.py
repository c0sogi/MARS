import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.signal import savgol_filter, welch
from scipy.stats import skew, kurtosis
import torch
import torchaudio

# ==========================================
# Configuration & Hyperparameters
# ==========================================

# Paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
CACHE_DIR = "./working/idea_17"

# Signal Processing Parameters
SAVGOL_WINDOW = 51
SAVGOL_POLY = 3
MFCC_N_MFCC = 5
SAMPLE_RATE = 100  # 60001 samples / 600 seconds approx 100Hz
N_FFT = 1024
HOP_LENGTH = 512

# Model Hyperparameters
SEED = 42
NUM_LEAVES = 128
LEARNING_RATE = 0.005
N_ESTIMATORS = 6000
EARLY_STOPPING_ROUNDS = 200
OBJECTIVE = "regression_l1"  # MAE
METRIC = "mae"
VERBOSITY = -1

# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================
# Feature Engineering
# ==========================================


def get_mfcc(signal_np, sr=SAMPLE_RATE, n_mfcc=MFCC_N_MFCC):
    """
    Compute MFCCs using Torchaudio for a 1D numpy signal.
    """
    # Ensure input is long enough for FFT
    if len(signal_np) < N_FFT:
        return np.zeros(n_mfcc)

    tensor = torch.from_numpy(signal_np).float()

    # Define MFCC transform
    transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": 23,
            "center": False,
        },
    )

    # Compute MFCCs: Output shape (n_mfcc, time)
    mfcc = transform(tensor)

    # Return mean over time dimension to get a fixed-size vector
    return torch.mean(mfcc, dim=1).numpy()


def extract_features_for_segment(file_path):
    """
    Implements the Augmented Orthogonal Decomposition pipeline.
    Returns a dictionary of features for a single data segment.
    """
    try:
        # Load data, using float32 to save memory
        df = pd.read_csv(file_path, dtype="float32")
    except FileNotFoundError:
        return None

    # Imputation: Fill NaNs with column mean (preserves DC offset)
    df = df.fillna(df.mean())

    sensor_cols = [c for c in df.columns if "sensor" in c]
    all_features = {}

    # Lists to store cross-sensor stats
    sensor_means = []
    sensor_rms = []

    for sensor in sensor_cols:
        raw = df[sensor].values

        # --- View A: Trend (Kinematics) ---
        # Savitzky-Golay Filter
        trend = savgol_filter(raw, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY)

        # Derivatives of Trend
        vel = np.gradient(trend)
        acc = np.gradient(vel)

        # Kinematic Features
        all_features[f"{sensor}_trend_mean"] = np.mean(trend)
        all_features[f"{sensor}_trend_std"] = np.std(trend)
        all_features[f"{sensor}_vel_std"] = np.std(vel)
        all_features[f"{sensor}_acc_std"] = np.std(acc)
        all_features[f"{sensor}_trend_q05"] = np.quantile(trend, 0.05)
        all_features[f"{sensor}_trend_q95"] = np.quantile(trend, 0.95)

        # --- View B: Texture (Timbre) ---
        # Residuals
        residual = raw - trend

        # MFCCs (Cepstral Features)
        mfccs = get_mfcc(residual)
        for i, val in enumerate(mfccs):
            all_features[f"{sensor}_res_mfcc_{i}"] = val

        # Higher-Order Moments
        all_features[f"{sensor}_res_skew"] = skew(residual)
        all_features[f"{sensor}_res_kurt"] = kurtosis(residual)

        # --- View C: Energy / Spectral ---
        # Raw Extremes
        all_features[f"{sensor}_raw_min"] = np.min(raw)
        all_features[f"{sensor}_raw_max"] = np.max(raw)
        all_features[f"{sensor}_raw_ptp"] = np.ptp(raw)

        # Spectral Density (Welch)
        freqs, psd = welch(raw, fs=SAMPLE_RATE, nperseg=1024)
        # Band Powers
        all_features[f"{sensor}_psd_low"] = np.sum(psd[freqs < 2])
        all_features[f"{sensor}_psd_mid"] = np.sum(psd[(freqs >= 2) & (freqs < 10)])
        all_features[f"{sensor}_psd_high"] = np.sum(psd[freqs >= 10])

        # Temporal Windows (Flattened)
        # Split into 10 non-overlapping windows
        chunks = np.array_split(raw, 10)
        chunk_rms_list = []
        for i, chunk in enumerate(chunks):
            c_mean = np.mean(chunk)
            c_rms = np.sqrt(np.mean(chunk**2))
            all_features[f"{sensor}_w{i}_mean"] = c_mean
            all_features[f"{sensor}_w{i}_rms"] = c_rms
            chunk_rms_list.append(c_rms)

        # Collect for spatial consistency
        sensor_means.append(np.mean(raw))
        sensor_rms.append(np.mean(chunk_rms_list))

    # --- Spatial Consistency Augmentation ---
    # Compute statistics across the sensor array
    all_features["spatial_mean_std"] = np.std(sensor_means)
    all_features["spatial_mean_range"] = np.ptp(sensor_means)
    all_features["spatial_rms_std"] = np.std(sensor_rms)
    all_features["spatial_rms_mean"] = np.mean(sensor_rms)

    return all_features


def process_dataset(meta_path, cache_filename, load_cached_data=True, debug_size=None):
    """
    Loads metadata, extracts features for each segment, and handles caching.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, cache_filename)

    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing dataset from {meta_path}...")
    meta_df = pd.read_csv(meta_path)

    if debug_size is not None:
        print(f"Debug mode: processing first {debug_size} rows.")
        meta_df = meta_df.head(debug_size)

    features_list = []
    segment_ids = []
    targets = []

    total = len(meta_df)
    for idx, row in meta_df.iterrows():
        # Metadata file_path is relative to input dir (e.g., "train/123.csv")
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        feats = extract_features_for_segment(full_path)
        if feats is not None:
            features_list.append(feats)
            segment_ids.append(row["segment_id"])
            if "time_to_eruption" in row:
                targets.append(row["time_to_eruption"])

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total} files")

    # Create DataFrame
    df_features = pd.DataFrame(features_list)
    df_features["segment_id"] = segment_ids
    if targets:
        df_features["time_to_eruption"] = targets

    # Save to cache
    print(f"Saving features to {cache_path}...")
    df_features.to_parquet(cache_path, index=False)

    return df_features


# ==========================================
# Training & Prediction
# ==========================================


def train_model(train_df, val_df, n_estimators=N_ESTIMATORS):
    """
    Trains a LightGBM regressor with early stopping.
    """
    # Exclude non-feature columns
    feature_cols = [
        c for c in train_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["time_to_eruption"]
    X_val = val_df[feature_cols]
    y_val = val_df["time_to_eruption"]

    print(f"Starting training with {len(feature_cols)} features...")
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        objective=OBJECTIVE,
        metric=METRIC,
        verbosity=VERBOSITY,
        random_state=SEED,
        n_jobs=-1,
    )

    # Callbacks for early stopping and logging
    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(period=100),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=callbacks,
    )

    return model, feature_cols


def generate_submission(model, test_df, feature_cols):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating predictions...")
    X_test = test_df[feature_cols]
    preds = model.predict(X_test)

    submission = pd.DataFrame(
        {"segment_id": test_df["segment_id"], "time_to_eruption": preds}
    )

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


def run_pipeline(load_cached_data=True, debug=False):
    """
    Main execution function.
    """
    debug_size = 50 if debug else None

    # 1. Process Data
    train_df = process_dataset(
        TRAIN_META_PATH, "train_features.parquet", load_cached_data, debug_size
    )
    val_df = process_dataset(
        VAL_META_PATH, "val_features.parquet", load_cached_data, debug_size
    )
    test_df = process_dataset(
        TEST_META_PATH, "test_features.parquet", load_cached_data, debug_size
    )

    # 2. Train Model
    model, feature_cols = train_model(train_df, val_df)

    # 3. Generate Submission
    generate_submission(model, test_df, feature_cols)
