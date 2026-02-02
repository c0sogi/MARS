import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.signal import savgol_filter, welch
from scipy.stats import skew, kurtosis
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class Config:
    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_34"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # --- Signal Processing Parameters ---
    # Savitzky-Golay (Trend)
    SAVGOL_WINDOW = 51
    SAVGOL_POLY = 2

    # Wavelet (Texture)
    WAVELET_NAME = "db4"

    # Spectral (Welch)
    WELCH_NPERSEG = 1024
    FREQ_BANDS = {"low": (0.1, 3), "mid": (3, 10), "high": (10, 45)}

    # Differential Profiling
    NUM_WINDOWS = 10

    # --- Feature Engineering ---
    QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    # --- Model Hyperparameters ---
    SEED = 42
    N_FOLDS = 5
    NUM_LEAVES = 31
    LEARNING_RATE = 0.05
    N_ESTIMATORS = 2000
    EARLY_STOPPING_ROUNDS = 100
    OBJECTIVE = "regression_l2"  # MSE for optimization
    METRIC = "l1"  # MAE for evaluation
    LAMBDA_L1 = 1.0
    LAMBDA_L2 = 1.0
    FEATURE_FRACTION = 0.7
    BAGGING_FRACTION = 0.7
    BAGGING_FREQ = 1
    VERBOSITY = -1


def calculate_entropy(x):
    """Calculates Shannon entropy of the energy distribution of signal x."""
    energy = np.sum(x**2)
    if energy == 0:
        return 0
    p = (x**2) / energy
    # Filter zeros to avoid log(0)
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def extract_features_for_sensor(signal, sensor_id, cfg):
    """
    Extracts features for a single sensor signal using the Hybrid Decomposition strategy.
    """
    features = {}
    prefix = f"{sensor_id}_"

    # 1. Preprocessing: Impute NaNs with mean
    if np.isnan(signal).any():
        signal = np.nan_to_num(signal, nan=np.nanmean(signal))

    # 2. Decomposition
    # View A: Trend (Savitzky-Golay)
    trend = savgol_filter(
        signal, window_length=cfg.SAVGOL_WINDOW, polyorder=cfg.SAVGOL_POLY
    )

    # View B: Texture (Wavelet Residuals)
    # We use the detail coefficients from level 1 decomposition
    # Fallback since pywt is not available
    texture = signal - trend

    # View C: Raw
    raw = signal

    # 3. Feature Engineering

    # --- From View A (Trend): Shape via Quantiles ---
    for q in cfg.QUANTILES:
        features[f"{prefix}trend_q{int(q*100)}"] = np.quantile(trend, q)

    # --- From View A (Trend): Kinematics (Derivatives) ---
    # Velocity
    vel = np.diff(trend)
    features[f"{prefix}vel_mean"] = np.mean(vel)
    features[f"{prefix}vel_std"] = np.std(vel)
    features[f"{prefix}vel_skew"] = skew(vel)
    features[f"{prefix}vel_kurt"] = kurtosis(vel)

    # Acceleration
    acc = np.diff(vel)
    features[f"{prefix}acc_mean"] = np.mean(acc)
    features[f"{prefix}acc_std"] = np.std(acc)
    features[f"{prefix}acc_skew"] = skew(acc)
    features[f"{prefix}acc_kurt"] = kurtosis(acc)

    # --- From View B (Texture): Wavelet Stats ---
    features[f"{prefix}txt_energy"] = np.sum(texture**2)
    features[f"{prefix}txt_entropy"] = calculate_entropy(texture)
    features[f"{prefix}txt_skew"] = skew(texture)
    features[f"{prefix}txt_kurt"] = kurtosis(texture)

    # --- From View C (Raw): Absolute Intensity ---
    features[f"{prefix}raw_min"] = np.min(raw)
    features[f"{prefix}raw_max"] = np.max(raw)
    features[f"{prefix}raw_ptp"] = np.ptp(raw)

    # --- From View C (Raw): Spectral Structure (Welch) ---
    freqs, psd = welch(raw, nperseg=cfg.WELCH_NPERSEG)
    for band_name, (low, high) in cfg.FREQ_BANDS.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        features[f"{prefix}spec_{band_name}"] = np.sum(psd[idx])

    # --- From View C (Raw): Differential Temporal Profiling ---
    # Split into N windows
    window_size = len(raw) // cfg.NUM_WINDOWS
    rms_values = []
    for i in range(cfg.NUM_WINDOWS):
        start = i * window_size
        end = (i + 1) * window_size if i < cfg.NUM_WINDOWS - 1 else len(raw)
        segment = raw[start:end]

        # Snapshot features
        rms = np.sqrt(np.mean(segment**2))
        rms_values.append(rms)
        features[f"{prefix}win{i}_rms"] = rms

    # Differential Dynamics (Ramping)
    for i in range(1, cfg.NUM_WINDOWS):
        features[f"{prefix}diff_win{i}_{i-1}"] = rms_values[i] - rms_values[i - 1]

    return features


def process_segment(file_path, segment_id, cfg):
    """Loads a CSV and extracts features for all 10 sensors."""
    try:
        df = pd.read_csv(file_path, dtype="float32")

        all_features = {}
        all_features["segment_id"] = int(segment_id)

        # Iterate through sensors 1 to 10
        for i in range(1, 11):
            col_name = f"sensor_{i}"
            if col_name in df.columns:
                sensor_feats = extract_features_for_sensor(
                    df[col_name].values, col_name, cfg
                )
                all_features.update(sensor_feats)
            else:
                # Handle missing sensor column if necessary (though data desc says no NaNs in columns usually)
                pass

        return all_features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def get_dataset(metadata_path, cfg, load_cached_data=True, dataset_name="train"):
    """
    Loads dataset from metadata. Uses caching to avoid re-processing.
    """
    os.makedirs(cfg.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(cfg.WORKING_DIR, f"{dataset_name}_features.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {dataset_name} data from {cache_path}...")
        df = pd.read_parquet(cache_path)

        # Separate X and y
        if "time_to_eruption" in df.columns:
            y = df["time_to_eruption"]
            X = df.drop(columns=["segment_id", "time_to_eruption"])
            return X, y, df["segment_id"]
        else:
            X = df.drop(columns=["segment_id"])
            return X, None, df["segment_id"]

    # 2. Process from Scratch
    print(f"Processing {dataset_name} data from scratch...")
    meta_df = pd.read_csv(metadata_path)

    feature_list = []

    # Iterate over metadata
    total = len(meta_df)
    for idx, row in meta_df.iterrows():
        full_path = os.path.join(cfg.INPUT_DIR, row["file_path"])
        feats = process_segment(full_path, row["segment_id"], cfg)

        if feats is not None:
            if "time_to_eruption" in row:
                feats["time_to_eruption"] = row["time_to_eruption"]
            feature_list.append(feats)

        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{total} files")

    df = pd.DataFrame(feature_list)

    # Save to cache
    print(f"Saving {dataset_name} data to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    if "time_to_eruption" in df.columns:
        y = df["time_to_eruption"]
        X = df.drop(columns=["segment_id", "time_to_eruption"])
        return X, y, df["segment_id"]
    else:
        X = df.drop(columns=["segment_id"])
        return X, None, df["segment_id"]


def train_model(X_train, y_train, X_val, y_val, cfg):
    """Trains a LightGBM model with early stopping."""
    print("Initializing LightGBM Dataset...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "num_leaves": cfg.NUM_LEAVES,
        "learning_rate": cfg.LEARNING_RATE,
        "objective": cfg.OBJECTIVE,
        "metric": cfg.METRIC,
        "feature_fraction": cfg.FEATURE_FRACTION,
        "bagging_fraction": cfg.BAGGING_FRACTION,
        "bagging_freq": cfg.BAGGING_FREQ,
        "verbosity": cfg.VERBOSITY,
        "seed": cfg.SEED,
        "n_jobs": -1,
    }

    print(
        f"Training LightGBM model (Leaves={cfg.NUM_LEAVES}, LR={cfg.LEARNING_RATE})..."
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=cfg.EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=cfg.N_ESTIMATORS,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    return model


def generate_predictions(model, X_test, test_ids, cfg):
    """Generates predictions and saves submission file."""
    print("Generating predictions for test set...")
    preds = model.predict(X_test, num_iteration=model.best_iteration)

    # Create submission DataFrame
    sub_df = pd.DataFrame({"segment_id": test_ids, "time_to_eruption": preds})

    os.makedirs(cfg.SUBMISSION_DIR, exist_ok=True)
    sub_df.to_csv(cfg.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {cfg.SUBMISSION_PATH}")


def run_pipeline(load_cached_data=True):
    """Main execution function."""
    cfg = Config()

    # 1. Load Data
    print("--- Loading Training Data ---")
    X_train, y_train, _ = get_dataset(
        cfg.TRAIN_METADATA, cfg, load_cached_data, "train"
    )

    print("--- Loading Validation Data ---")
    X_val, y_val, _ = get_dataset(cfg.VAL_METADATA, cfg, load_cached_data, "val")

    # 2. Train Model
    print("--- Starting Training ---")
    model = train_model(X_train, y_train, X_val, y_val, cfg)

    # 3. Inference
    print("--- Loading Test Data ---")
    X_test, _, test_ids = get_dataset(cfg.TEST_METADATA, cfg, load_cached_data, "test")

    print("--- Generating Submission ---")
    generate_predictions(model, X_test, test_ids, cfg)

    print("Pipeline completed successfully.")
