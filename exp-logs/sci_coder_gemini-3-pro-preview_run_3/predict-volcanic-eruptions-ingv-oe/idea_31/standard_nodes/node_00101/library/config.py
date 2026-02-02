import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed
import scipy.signal
import scipy.stats

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

# Paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_optimized"
SUBMISSION_DIR = "./submission"

# Signal Processing Parameters
FS = 100  # Sampling frequency (10 mins * 60s * 100Hz = 60000 samples)
SG_WINDOW = 51
SG_ORDER = 2
WELCH_NPERSEG = 1024
FREQ_BANDS = {"low": (0.1, 3), "mid": (3, 10), "high": (10, 45)}
NUM_TEMPORAL_WINDOWS = 10

# Model Hyperparameters
NUM_FOLDS = 5
SEED = 42
EARLY_STOPPING_ROUNDS = 100

LGBM_PARAMS = {
    "num_leaves": 128,
    "learning_rate": 0.01,
    "n_estimators": 10000,
    "objective": "regression_l2",
    "metric": "mae",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": 4,  # Restrict threads per model to allow potential parallel folding
    "random_state": SEED,
}

# ==========================================
# FEATURE ENGINEERING
# ==========================================


def extract_features(df):
    """
    Implements the Order-Constrained Pyramidal Decomposition with Differential Temporal Profiling.
    """
    # Impute missing values with column means (Segment-wise)
    df = df.fillna(df.mean())

    features = {}

    # Process each sensor
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    for sensor in sensor_cols:
        x = df[sensor].values.astype(np.float32)

        # --- View A: Trend (Savitzky-Golay) ---
        # Constraint: Order 2 to avoid overfitting acceleration noise
        trend = scipy.signal.savgol_filter(
            x, window_length=SG_WINDOW, polyorder=SG_ORDER
        )

        # --- View B: Texture (Residuals) ---
        # Approximating DWT detail analysis using Residuals stats
        residual = x - trend

        # --- Feature Group 1: Kinematics (from View A) ---
        vel = np.diff(trend)
        acc = np.diff(vel)

        for name, sig in [("trend", trend), ("vel", vel), ("acc", acc)]:
            features[f"{sensor}_{name}_mean"] = np.mean(sig)
            features[f"{sensor}_{name}_std"] = np.std(sig)
            features[f"{sensor}_{name}_skew"] = float(scipy.stats.skew(sig))
            features[f"{sensor}_{name}_kurt"] = float(scipy.stats.kurtosis(sig))

        # --- Feature Group 2: Texture Stats (from View B) ---
        features[f"{sensor}_res_energy"] = np.sum(residual**2)
        features[f"{sensor}_res_skew"] = float(scipy.stats.skew(residual))
        features[f"{sensor}_res_kurt"] = float(scipy.stats.kurtosis(residual))

        # Entropy of energy distribution
        p = residual**2
        p_sum = np.sum(p)
        if p_sum > 0:
            p_norm = p / p_sum
            features[f"{sensor}_res_entropy"] = -np.sum(p_norm * np.log(p_norm + 1e-12))
        else:
            features[f"{sensor}_res_entropy"] = 0.0

        # --- Feature Group 3: Intensity (from View C - Raw) ---
        features[f"{sensor}_min"] = np.min(x)
        features[f"{sensor}_max"] = np.max(x)
        features[f"{sensor}_ptp"] = np.ptp(x)  # Peak-to-Peak

        # --- Feature Group 4: High-Res Spectral (from View C) ---
        # Welch's Method with high nperseg for better low-freq resolution
        f, Pxx = scipy.signal.welch(x, fs=FS, nperseg=WELCH_NPERSEG)

        for band, (low, high) in FREQ_BANDS.items():
            mask = (f >= low) & (f <= high)
            if np.any(mask):
                features[f"{sensor}_spec_{band}"] = np.trapz(Pxx[mask], f[mask])
            else:
                features[f"{sensor}_spec_{band}"] = 0.0

        # --- Feature Group 5: Differential Temporal Profiling (from View C) ---
        # Split signal into N windows
        windows = np.array_split(x, NUM_TEMPORAL_WINDOWS)
        rms_vals = []
        mean_vals = []

        # 5a. Snapshots
        for i, w in enumerate(windows):
            rms = np.sqrt(np.mean(w**2))
            mean_val = np.mean(w)
            rms_vals.append(rms)
            mean_vals.append(mean_val)

            features[f"{sensor}_win{i}_rms"] = rms
            features[f"{sensor}_win{i}_mean"] = mean_val

        # 5b. Differential Dynamics (Gradient of the envelope)
        rms_diff = np.diff(rms_vals)
        mean_diff = np.diff(mean_vals)

        for i, d in enumerate(rms_diff):
            features[f"{sensor}_diff_rms_{i}"] = d
        for i, d in enumerate(mean_diff):
            features[f"{sensor}_diff_mean_{i}"] = d

    return features


def _process_file(row):
    """Helper for parallel processing."""
    try:
        file_path = os.path.join(INPUT_DIR, row["file_path"])
        df = pd.read_csv(file_path, dtype="float32")
        feats = extract_features(df)
        feats["segment_id"] = int(row["segment_id"])
        if "time_to_eruption" in row:
            feats["time_to_eruption"] = row["time_to_eruption"]
        return feats
    except Exception as e:
        print(f"Error processing {row.get('segment_id', 'unknown')}: {e}")
        return None


# ==========================================
# DATA PROCESSING PIPELINE
# ==========================================


def process_data(metadata_path, cache_name, load_cached_data=True):
    """
    Loads metadata, processes sensor files in parallel, and handles caching.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(WORKING_DIR, cache_name)

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Compute if cache missing or forced reload
    print(f"Processing data from {metadata_path}...")
    meta_df = pd.read_csv(metadata_path)

    # Parallel execution
    results = Parallel(n_jobs=12, verbose=0)(
        delayed(_process_file)(row) for _, row in meta_df.iterrows()
    )

    # Filter failures
    results = [r for r in results if r is not None]
    df_features = pd.DataFrame(results)

    # 3. Save to cache
    print(f"Saving features to {cache_path}")
    df_features.to_parquet(cache_path, index=False)

    return df_features


# ==========================================
# MODEL TRAINING
# ==========================================


def train_model(train_df):
    """
    Trains a Homogeneous Ensemble of LightGBM Regressors using Stratified K-Fold.
    """
    X = train_df.drop(columns=["segment_id", "time_to_eruption"])
    y = train_df["time_to_eruption"]

    # Stratification bins for continuous target
    num_bins = 10
    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    models = []
    oof_preds = np.zeros(len(X))
    scores = []

    print(f"Starting training with {NUM_FOLDS} folds on {len(X)} samples...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),  # Silent
        ]

        model = lgb.train(
            LGBM_PARAMS, train_set, valid_sets=[train_set, val_set], callbacks=callbacks
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred

        mae = mean_absolute_error(y_val, val_pred)
        scores.append(mae)
        models.append(model)

        print(f"Fold {fold + 1} MAE: {mae:.6f}")

    print(f"Average CV MAE: {np.mean(scores):.6f}")
    return models


# ==========================================
# SUBMISSION
# ==========================================


def generate_submission(models, test_df):
    """
    Generates predictions using the ensemble and saves to CSV.
    """
    X_test = test_df.drop(columns=["segment_id"])
    segment_ids = test_df["segment_id"]

    final_preds = np.zeros(len(X_test))

    # Average predictions across all fold models
    for model in models:
        preds = model.predict(X_test, num_iteration=model.best_iteration)
        final_preds += preds

    final_preds /= len(models)

    submission = pd.DataFrame(
        {"segment_id": segment_ids, "time_to_eruption": final_preds}
    )

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


# ==========================================
# ORCHESTRATION
# ==========================================


def run_pipeline():
    """
    Main execution function to be called by the user.
    """
    # 1. Process Data
    # Combine train and val metadata to use all labeled data for CV
    train_feats = process_data(
        os.path.join(METADATA_DIR, "train.csv"), "train_features.parquet"
    )
    val_feats = process_data(
        os.path.join(METADATA_DIR, "val.csv"), "val_features.parquet"
    )
    test_feats = process_data(
        os.path.join(METADATA_DIR, "test.csv"), "test_features.parquet"
    )

    full_train_df = pd.concat([train_feats, val_feats], axis=0).reset_index(drop=True)

    # 2. Train
    models = train_model(full_train_df)

    # 3. Predict & Submit
    generate_submission(models, test_feats)
