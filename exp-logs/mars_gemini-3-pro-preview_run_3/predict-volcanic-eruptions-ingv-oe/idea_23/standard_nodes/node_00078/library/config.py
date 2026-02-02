import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import signal, stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_24"
    SUBMISSION_DIR = "./submission"

    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Data
    SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]
    SEED = 42

    # Feature Extraction
    SG_WINDOW_SIZE = 51
    SG_POLY_ORDER = 2
    QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    NUM_TEMPORAL_WINDOWS = 10

    # Model (LightGBM High-Capacity)
    N_FOLDS = 5
    MODEL_PARAMS = {
        "objective": "regression_l2",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 128,
        "learning_rate": 0.02,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_estimators": 10000,
        "early_stopping_rounds": 100,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": 12,  # Use available vCPUs
    }


def extract_features(df):
    """
    Implements the Hybrid-Transform Decomposition and Dense Quantile Profiling.
    """
    # 1. Imputation: Fill NaNs with column means to preserve DC offsets
    # Fallback to 0 if column is all-NaN (Cite debug_lesson_1)
    df = df.fillna(df.mean()).fillna(0)

    features = {}

    for sensor in Config.SENSOR_COLS:
        if sensor not in df.columns:
            continue

        raw = df[sensor].values.astype(np.float32)

        # --- View A: Trend (Savitzky-Golay) ---
        # Isolates low-frequency baseline drift
        try:
            trend = signal.savgol_filter(
                raw, window_length=Config.SG_WINDOW_SIZE, polyorder=Config.SG_POLY_ORDER
            )
        except Exception:
            trend = raw  # Fallback if signal too short

        # Kinematics
        vel = np.gradient(trend)
        acc = np.gradient(vel)

        # Dense Quantiles on Kinematics
        for q in Config.QUANTILES:
            features[f"{sensor}_trend_q{int(q*100)}"] = np.quantile(trend, q)
            features[f"{sensor}_vel_q{int(q*100)}"] = np.quantile(vel, q)
            features[f"{sensor}_acc_q{int(q*100)}"] = np.quantile(acc, q)

        # --- View B: Texture (Residuals) ---
        # Residuals = Raw - Trend
        resid = raw - trend

        # Energy and Entropy of Residuals (Proxy for Wavelet Texture if pywt is missing)
        features[f"{sensor}_resid_rms"] = np.sqrt(np.mean(resid**2))

        # Histogram Entropy
        hist_counts, _ = np.histogram(resid, bins=50, density=True)
        features[f"{sensor}_resid_entropy"] = stats.entropy(hist_counts + 1e-10)

        # --- View C: Raw Signal Analysis ---
        # Absolute Intensity
        features[f"{sensor}_min"] = np.min(raw)
        features[f"{sensor}_max"] = np.max(raw)
        features[f"{sensor}_ptp"] = np.ptp(raw)

        # Spectral Structure (Welch PSD)
        # fs = 100Hz (60000 samples in 10 mins)
        f, Pxx = signal.welch(raw, fs=100, nperseg=256)

        # Band Power: Low (0-5Hz), Mid (5-15Hz), High (15-50Hz)
        features[f"{sensor}_band_low"] = np.sum(Pxx[(f >= 0) & (f < 5)])
        features[f"{sensor}_band_mid"] = np.sum(Pxx[(f >= 5) & (f < 15)])
        features[f"{sensor}_band_high"] = np.sum(Pxx[(f >= 15) & (f <= 50)])

        # Temporal Evolution (Split into windows)
        windows = np.array_split(raw, Config.NUM_TEMPORAL_WINDOWS)
        for i, w in enumerate(windows):
            features[f"{sensor}_w{i}_rms"] = np.sqrt(np.mean(w**2))

    return features


def load_and_process_data(meta_path, load_cached_data=True, is_test=False):
    """
    Loads metadata, processes features with caching, and returns DataFrame.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache path
    name = os.path.basename(meta_path).replace(".csv", "")
    cache_path = os.path.join(Config.WORKING_DIR, f"{name}_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing data from {meta_path}...")
    meta_df = pd.read_csv(meta_path)

    # For debugging/demo, we might limit size, but for full run we use all
    # meta_df = meta_df.head(100)

    feature_list = []

    for idx, row in meta_df.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            # Load sensor data
            df = pd.read_csv(file_path, dtype="float32")

            # Extract features
            feats = extract_features(df)
            feats["segment_id"] = int(row["segment_id"])

            if not is_test:
                feats["time_to_eruption"] = row["time_to_eruption"]

            feature_list.append(feats)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Create DataFrame
    features_df = pd.DataFrame(feature_list)

    # Save to cache
    features_df.to_parquet(cache_path, index=False)
    print(f"Saved features to {cache_path}")

    return features_df


def train_ensemble(train_df):
    """
    Trains a homogeneous ensemble of High-Capacity LightGBM models using Stratified K-Fold.
    """
    print("Starting Ensemble Training...")

    feature_cols = [
        c for c in train_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    X = train_df[feature_cols]
    y = train_df["time_to_eruption"]

    # Create bins for stratification (emulating the metadata strategy)
    num_bins = 10
    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    models = []
    oof_preds = np.zeros(len(train_df))
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.MODEL_PARAMS["early_stopping_rounds"],
                verbose=False,
            ),
            lgb.log_evaluation(period=1000),
        ]

        model = lgb.train(
            Config.MODEL_PARAMS,
            train_set,
            valid_sets=[train_set, val_set],
            callbacks=callbacks,
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred

        score = mean_absolute_error(y_val, val_pred)
        scores.append(score)
        models.append(model)

        print(f"Fold {fold+1} MAE: {score:.4f}")

    print(f"Average CV MAE: {np.mean(scores):.4f}")
    return models


def predict_and_submit(models, test_df):
    """
    Generates predictions using the ensemble and saves submission.
    """
    print("Generating predictions for test set...")

    feature_cols = [
        c for c in test_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    X_test = test_df[feature_cols]

    # Ensemble prediction
    preds = np.zeros(len(X_test))
    for model in models:
        preds += model.predict(X_test, num_iteration=model.best_iteration)

    preds /= len(models)

    # Prepare submission
    submission = pd.DataFrame(
        {"segment_id": test_df["segment_id"], "time_to_eruption": preds}
    )

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())


def main():
    # 1. Load Train Data
    # Note: We use the provided train.csv metadata.
    # We could also merge val.csv if we wanted full data, but we'll stick to train.csv for CV stability.
    train_df = load_and_process_data(
        Config.TRAIN_META_PATH, load_cached_data=True, is_test=False
    )

    # 2. Train Ensemble
    models = train_ensemble(train_df)

    # 3. Load Test Data
    test_df = load_and_process_data(
        Config.TEST_META_PATH, load_cached_data=True, is_test=True
    )

    # 4. Predict and Submit
    predict_and_submit(models, test_df)


# Execute the pipeline
if __name__ == "__main__":
    main()
