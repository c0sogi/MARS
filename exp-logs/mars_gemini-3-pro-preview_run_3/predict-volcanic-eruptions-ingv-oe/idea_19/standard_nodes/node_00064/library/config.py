import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.signal import welch, savgol_filter
from scipy.stats import entropy, kurtosis, skew
import warnings

# --- Global Configuration ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_20"
SUBMISSION_DIR = "./submission"
SEED = 42
NUM_SENSORS = 10

# --- Feature Extraction Hyperparameters ---
# Savitzky-Golay Settings for Trend Decomposition
SG_WINDOW = 51
SG_POLY = 2

# Dense Quantile Grid for capturing distribution shape
QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# Spectral Bands (Normalized Frequency assuming fs=100Hz)
# Low: 0-5Hz, Mid: 5-20Hz, High: 20-50Hz
PSD_BANDS = {"low": (0, 5), "mid": (5, 20), "high": (20, 50)}
SAMPLING_RATE = 100  # 60001 samples in 10 mins = 100 Hz
NUM_WINDOWS = 10  # For flattened temporal features

# --- Model Hyperparameters ---
EARLY_STOPPING_ROUNDS = 100
LGBM_PARAMS = {
    "objective": "regression",  # L2 Loss for stable gradient descent
    "metric": "mae",  # Evaluation metric
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 128,  # High capacity for dense features
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 10000,
    "n_jobs": 12,
    "verbosity": -1,
    "seed": SEED,
    "device": "cpu",  # Default to CPU; change to 'gpu' if compatible
}

# Set global seed
np.random.seed(SEED)

# --- Core Functions ---


def load_data(file_path):
    """
    Loads sensor data from CSV with memory optimization.
    """
    full_path = os.path.join(INPUT_DIR, file_path)
    # Use float32 to handle NaNs and reduce memory usage
    df = pd.read_csv(full_path, dtype="float32")
    return df


def extract_features(df):
    """
    Implements Trend/Texture Decomposition with Shift-Invariant Temporal Features.
    Cite solution_lesson_node_00049: Decompose Time-Series into Trend and Texture.
    Cite solution_lesson_node_00050: Shift Invariance vs. Temporal Specificity.
    """
    # 1. Imputation: Fill NaNs with sensor mean to preserve DC offset
    df = df.fillna(df.mean())

    features = {}

    for sensor in df.columns:
        sig = df[sensor].values

        # --- View 1: Trend/Texture Decomposition (Cite solution_lesson_node_00049) ---
        # Trend via Savitzky-Golay (Low Frequency)
        trend = savgol_filter(sig, window_length=SG_WINDOW, polyorder=SG_POLY)
        # Texture (High Frequency Residuals)
        texture = sig - trend

        # --- View 2: Kinematics on Trend (Cite solution_lesson_node_00021) ---
        vel = np.diff(trend)
        acc = np.diff(vel)

        # Granular Quantiles on Trend Kinematics (Cite solution_lesson_node_00059)
        for q in QUANTILES:
            features[f"{sensor}_trend_q{q}"] = np.quantile(trend, q)
            features[f"{sensor}_vel_q{q}"] = np.quantile(vel, q)
            features[f"{sensor}_acc_q{q}"] = np.quantile(acc, q)

        # --- View 3: Texture Statistics (Cite solution_lesson_node_00049) ---
        features[f"{sensor}_text_rms"] = np.sqrt(np.mean(texture**2))
        features[f"{sensor}_text_kurt"] = kurtosis(texture)
        features[f"{sensor}_text_skew"] = skew(texture)

        # --- View 4: Shift-Invariant Temporal Features (Cite solution_lesson_node_00050) ---
        # Split into windows
        n_samples = len(sig)
        n_trim = n_samples - (n_samples % NUM_WINDOWS)
        sig_reshaped = sig[:n_trim].reshape(NUM_WINDOWS, -1)

        # Compute stats per window
        w_means = np.mean(sig_reshaped, axis=1)
        w_stds = np.std(sig_reshaped, axis=1)
        w_rms = np.sqrt(np.mean(sig_reshaped**2, axis=1))

        # Aggregate window stats (Shift Invariance)
        # Instead of flattening (w0, w1...), we summarize the distribution of window stats
        features[f"{sensor}_win_mean_std"] = np.std(w_means)
        features[f"{sensor}_win_std_mean"] = np.mean(w_stds)
        features[f"{sensor}_win_rms_mean"] = np.mean(w_rms)
        features[f"{sensor}_win_rms_std"] = np.std(w_rms)
        features[f"{sensor}_win_mean_min"] = np.min(w_means)
        features[f"{sensor}_win_mean_max"] = np.max(w_means)

        # --- View 5: Structural Spectral Features (Cite solution_lesson_node_00007) ---
        # Using Welch's method (Cite solution_lesson_node_00054)
        f, Pxx = welch(sig, fs=SAMPLING_RATE, nperseg=256)
        for band_name, (low_f, high_f) in PSD_BANDS.items():
            idx = np.logical_and(f >= low_f, f <= high_f)
            if np.sum(idx) > 0:
                features[f"{sensor}_psd_{band_name}"] = np.sum(Pxx[idx])
            else:
                features[f"{sensor}_psd_{band_name}"] = 0.0

        # Raw Extrema (Cite solution_lesson_node_00031)
        features[f"{sensor}_min"] = np.min(sig)
        features[f"{sensor}_max"] = np.max(sig)

    return features


def process_dataset(
    metadata_path, output_filename, load_cached_data=True, debug_size=None
):
    """
    Orchestrates data loading, feature extraction, and caching.

    Args:
        metadata_path (str): Path to the metadata CSV.
        output_filename (str): Name of the parquet file to save/load.
        load_cached_data (bool): If True, attempts to load from disk.
        debug_size (int, optional): If set, only process N samples.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(WORKING_DIR, output_filename)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Process from Scratch
    print(f"Processing data from {metadata_path}...")
    meta_df = pd.read_csv(metadata_path)

    if debug_size:
        meta_df = meta_df.iloc[:debug_size]
        print(f"Debug mode: processing {debug_size} samples.")

    data_list = []

    for i, row in meta_df.iterrows():
        segment_id = row["segment_id"]
        file_path = row["file_path"]

        try:
            df_sensor = load_data(file_path)
            feats = extract_features(df_sensor)
            feats["segment_id"] = segment_id

            # Attach target if it exists (Train/Val sets)
            if "time_to_eruption" in row:
                feats["time_to_eruption"] = row["time_to_eruption"]

            data_list.append(feats)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    result_df = pd.DataFrame(data_list)

    # 3. Save Cache (only if not debugging)
    if not debug_size:
        result_df.to_parquet(cache_path, index=False)
        print(f"Saved features to {cache_path}")

    return result_df


def train_model(X_train, y_train, X_val, y_val):
    """
    Trains the LightGBM Regressor with Early Stopping.
    """
    print(
        f"Starting LightGBM training. Train shape: {X_train.shape}, Val shape: {X_val.shape}"
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    return model


def generate_submission(model, test_df, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Identify feature columns (exclude IDs and Targets)
    feature_cols = [
        c for c in test_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    X_test = test_df[feature_cols]

    print(f"Predicting on {len(X_test)} test samples...")
    preds = model.predict(X_test, num_iteration=model.best_iteration)

    submission = pd.DataFrame(
        {"segment_id": test_df["segment_id"], "time_to_eruption": preds}
    )

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
