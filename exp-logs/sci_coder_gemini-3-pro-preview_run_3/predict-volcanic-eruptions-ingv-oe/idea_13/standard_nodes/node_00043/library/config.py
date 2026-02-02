import os
import glob
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.stats import iqr
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration Constants ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_13")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Global Constants
SEED = 42
N_FOLDS = 5
N_JOBS = 12  # Use available vCPUs

# Signal Processing Parameters
SAVGOL_WINDOW = 25
SAVGOL_POLYORDER = 2
NUM_TEMPORAL_WINDOWS = 10
SAMPLING_RATE = (
    100.0  # Assumed based on typical seismic data (10 mins = 600s, 60k rows -> 100Hz)
)

# Model Hyperparameters

# LightGBM
LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 5000,
    "verbosity": -1,
    "random_state": SEED,
    "n_jobs": N_JOBS,
}

# XGBoost
XGB_PARAMS = {
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 5000,
    "random_state": SEED,
    "n_jobs": N_JOBS,
    "tree_method": "hist",
}

# HistGradientBoosting (Replacement for CatBoost due to dependency constraints)
HGB_PARAMS = {
    "loss": "absolute_error",
    "learning_rate": 0.05,
    "max_iter": 5000,
    "max_depth": 6,
    "l2_regularization": 3.0,
    "random_state": SEED,
    "early_stopping": True,
    "n_iter_no_change": 100,
    "validation_fraction": 0.1,
}

# Ridge Regression (Meta Learner)
RIDGE_PARAMS = {"alpha": 10.0, "random_state": SEED}

# --- Feature Extraction Logic ---


def process_segment(df):
    """
    Implements the Dual-Stream Kinematic-Extremum feature extraction.
    Args:
        df: DataFrame (60001, 10) of sensor readings.
    Returns:
        1D numpy array of features.
    """
    # 1. Imputation (Stream A - Raw)
    # Fill NaNs with column mean
    df = df.fillna(df.mean())

    # Convert to numpy for speed
    raw_data = df.values  # (60001, 10)
    n_sensors = raw_data.shape[1]

    features = []

    # --- Per-Sensor Features ---
    for i in range(n_sensors):
        x_raw = raw_data[:, i]

        # 2. Stream B: Smoothing
        x_smooth = signal.savgol_filter(
            x_raw, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER
        )

        # --- View 1: Raw Intensity (Stream A) ---
        features.append(np.min(x_raw))
        features.append(np.max(x_raw))
        features.append(np.ptp(x_raw))

        # --- View 2: Robust Kinematics (Stream B) ---
        # Velocity (1st Derivative)
        vel = np.gradient(x_smooth)
        features.extend(np.quantile(vel, [0.01, 0.05, 0.95, 0.99]))
        features.append(np.mean(vel))
        features.append(np.std(vel))

        # Acceleration (2nd Derivative)
        acc = np.gradient(vel)
        features.extend(np.quantile(acc, [0.01, 0.05, 0.95, 0.99]))
        features.append(np.mean(acc))
        features.append(np.std(acc))

        # --- View 4: Structural Spectral Features (Stream A) ---
        # PSD using Welch's method
        freqs, psd = signal.welch(x_raw, fs=SAMPLING_RATE, nperseg=1024)

        # Band Powers
        # Low (0-2Hz), Mid (2-10Hz), High (10-20Hz) - assuming 100Hz Nyquist is 50
        feat_psd_low = np.sum(psd[(freqs >= 0) & (freqs < 2)])
        feat_psd_mid = np.sum(psd[(freqs >= 2) & (freqs < 10)])
        feat_psd_high = np.sum(psd[(freqs >= 10) & (freqs < 20)])
        features.extend([feat_psd_low, feat_psd_mid, feat_psd_high])

        # Spectral Centroid
        spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-9)
        features.append(spectral_centroid)

        # --- View 5: Flattened Temporal Windows (Stream A) ---
        # Split into N windows
        wins = np.array_split(x_raw, NUM_TEMPORAL_WINDOWS)
        for w in wins:
            features.append(np.sqrt(np.mean(w**2)))  # RMS
            features.append(np.mean(w))  # Mean

    # --- View 6: Spatial Fingerprinting (Stream B) ---
    # Pairwise correlations of smoothed signals
    # Create DataFrame for easy corr computation
    df_smooth = pd.DataFrame(
        np.array(
            [
                signal.savgol_filter(raw_data[:, i], SAVGOL_WINDOW, SAVGOL_POLYORDER)
                for i in range(n_sensors)
            ]
        ).T
    )
    corr_matrix = df_smooth.corr().abs()
    # Extract upper triangle
    upper_indices = np.triu_indices(n_sensors, k=1)
    corr_values = corr_matrix.values[upper_indices]
    features.extend(corr_values)

    return np.array(features, dtype=np.float32)


# --- Data Loading & Caching ---


def make_dataset(metadata_df, load_cached_data=True, debug_size=None):
    """
    Loads data, computes features, and handles caching.
    """
    if debug_size:
        metadata_df = metadata_df.iloc[:debug_size]
        print(f"Debug mode: processing {len(metadata_df)} samples.")

    X = []
    y = []
    segment_ids = []

    # Determine cache file path based on metadata hash or length
    # For simplicity, we use separate filenames for train/val/test based on dataframe content
    is_test = "time_to_eruption" not in metadata_df.columns
    cache_filename = "test_features.parquet" if is_test else "train_features.parquet"
    if debug_size:
        cache_filename = f"debug_{debug_size}_{cache_filename}"

    cache_path = os.path.join(CACHE_DIR, cache_filename)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        cached_df = pd.read_parquet(cache_path)
        # Align with metadata
        # We assume metadata order matches or we merge.
        # Let's merge on segment_id to be safe
        merged = metadata_df.merge(cached_df, on="segment_id", how="left")

        feature_cols = [c for c in cached_df.columns if c.startswith("f_")]
        X = merged[feature_cols].values
        segment_ids = merged["segment_id"].values
        if not is_test:
            y = merged["time_to_eruption"].values

        return X, y, segment_ids

    print(f"Computing features for {len(metadata_df)} files...")

    feature_names = None

    for idx, row in metadata_df.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])
        seg_id = row["segment_id"]

        try:
            df = pd.read_csv(file_path, dtype="float32")
            feats = process_segment(df)

            X.append(feats)
            segment_ids.append(seg_id)
            if not is_test:
                y.append(row["time_to_eruption"])

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # If error, append zeros or skip. Skipping might break alignment.
            # We'll append zeros matching the length of the last successful one
            if len(X) > 0:
                X.append(np.zeros_like(X[-1]))
            else:
                # Should not happen if first file works
                pass
            segment_ids.append(seg_id)
            if not is_test:
                y.append(row["time_to_eruption"])

    X = np.array(X)
    if not is_test:
        y = np.array(y)

    # Save to cache
    print(f"Saving features to {cache_path}...")
    feature_cols = [f"f_{i}" for i in range(X.shape[1])]
    cache_df = pd.DataFrame(X, columns=feature_cols)
    cache_df["segment_id"] = segment_ids
    cache_df.to_parquet(cache_path)

    return X, y, np.array(segment_ids)


# --- Training Logic ---


def train_stacking_ensemble(X, y, folds=N_FOLDS):
    """
    Trains Level 0 (LGBM, XGB, HGB) and Level 1 (Ridge) models.
    Returns trained models for inference.
    """
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    # Bin targets for stratified split
    num_bins = int(np.floor(1 + np.log2(len(y))))
    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    oof_preds = np.zeros((X.shape[0], 3))  # 3 Base models
    base_models_cv = []  # To store CV models if we wanted to use them for averaging

    # Store best iterations for retraining
    best_iters_lgbm = []
    best_iters_xgb = []

    print(f"Starting {folds}-Fold CV Training...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_bins)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # --- Model 1: LightGBM ---
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        model_lgb = lgb.train(
            LGBM_PARAMS,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),  # Silent
            ],
        )
        oof_preds[val_idx, 0] = model_lgb.predict(
            X_val, num_iteration=model_lgb.best_iteration
        )
        best_iters_lgbm.append(model_lgb.best_iteration)

        # --- Model 2: XGBoost ---
        model_xgb = xgb.XGBRegressor(**XGB_PARAMS)
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds[val_idx, 1] = model_xgb.predict(X_val)
        # XGBoost handles early stopping internally, best_iteration is accessible
        best_iters_xgb.append(model_xgb.best_iteration)

        # --- Model 3: HistGradientBoosting (Scikit-Learn) ---
        model_hgb = HistGradientBoostingRegressor(**HGB_PARAMS)
        model_hgb.fit(X_train, y_train)
        oof_preds[val_idx, 2] = model_hgb.predict(X_val)

        print(
            f"Fold {fold+1} MAE: LGB={mean_absolute_error(y_val, oof_preds[val_idx, 0]):.0f}, "
            f"XGB={mean_absolute_error(y_val, oof_preds[val_idx, 1]):.0f}, "
            f"HGB={mean_absolute_error(y_val, oof_preds[val_idx, 2]):.0f}"
        )

    # --- Level 1: Meta Learner ---
    print("Training Meta Learner (Ridge)...")
    meta_model = Ridge(**RIDGE_PARAMS)
    meta_model.fit(oof_preds, y)

    oof_meta_pred = meta_model.predict(oof_preds)
    final_cv_mae = mean_absolute_error(y, oof_meta_pred)
    print(f"Final Stacking CV MAE: {final_cv_mae:.4f}")

    # --- Retraining on Full Data ---
    print("Retraining Base Learners on Full Data...")

    # LGBM
    avg_iter_lgbm = int(np.mean(best_iters_lgbm))
    lgb_full = lgb.Dataset(X, y)
    final_lgbm = lgb.train(
        {**LGBM_PARAMS, "n_estimators": avg_iter_lgbm}, lgb_full  # Fix iterations
    )

    # XGB
    avg_iter_xgb = int(np.mean(best_iters_xgb))
    # XGBoost doesn't support setting n_estimators in fit easily with early_stopping off unless we re-init
    final_xgb_params = XGB_PARAMS.copy()
    final_xgb_params["n_estimators"] = avg_iter_xgb
    final_xgb_params["early_stopping_rounds"] = None  # Disable for full train
    final_xgb = xgb.XGBRegressor(**final_xgb_params)
    final_xgb.fit(X, y, verbose=False)

    # HGB
    final_hgb = HistGradientBoostingRegressor(**HGB_PARAMS)
    final_hgb.fit(X, y)

    return {"lgbm": final_lgbm, "xgb": final_xgb, "hgb": final_hgb, "meta": meta_model}


# --- Prediction & Submission ---


def generate_submission(test_df, models):
    """
    Generates predictions for test set and saves submission file.
    """
    print("Generating Test Predictions...")
    X_test, _, segment_ids = make_dataset(test_df, load_cached_data=True)

    # Base Predictions
    pred_lgbm = models["lgbm"].predict(
        X_test, num_iteration=models["lgbm"].best_iteration
    )
    pred_xgb = models["xgb"].predict(X_test)
    pred_hgb = models["hgb"].predict(X_test)

    # Stack
    base_preds = np.column_stack([pred_lgbm, pred_xgb, pred_hgb])
    final_preds = models["meta"].predict(base_preds)

    # Create Submission DataFrame
    sub_df = pd.DataFrame({"segment_id": segment_ids, "time_to_eruption": final_preds})

    # Save
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


# --- Main Execution Helper (Optional) ---
def run_pipeline():
    # Load Metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Load/Create Train Data
    X, y, _ = make_dataset(train_meta, load_cached_data=True)

    # Train
    models = train_stacking_ensemble(X, y)

    # Predict
    generate_submission(test_meta, models)
