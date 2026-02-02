import os
import glob
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/optimized_solution"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Signal Processing
SMOOTHING_WINDOW = 21
POLY_ORDER = 2

# Training
N_FOLDS = 5
RANDOM_SEED = 42
N_JOBS = 12

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# FEATURE ENGINEERING
# ==========================================


def extract_segment_features(file_path, segment_id):
    """
    Extracts Peak-Aware and Kinematic features for a single segment.
    """
    try:
        # Load data (float32 to handle NaNs and memory)
        full_path = os.path.join(INPUT_DIR, file_path)
        df = pd.read_csv(full_path, dtype="float32")

        # Impute missing values with column mean to preserve DC offset
        df = df.fillna(df.mean())

        features = {}
        features["segment_id"] = segment_id

        # Identify sensor columns
        sensor_cols = [c for c in df.columns if "sensor" in c]

        for col in sensor_cols:
            x = df[col].values

            # --- View 1: Raw Extrema (Correction for Peak Intensity) ---
            features[f"{col}_raw_min"] = np.min(x)
            features[f"{col}_raw_max"] = np.max(x)
            features[f"{col}_raw_ptp"] = np.ptp(x)

            # --- View 2: Smoothed Kinematics (Exploitation) ---
            # Savitzky-Golay Smoothing
            try:
                x_smooth = signal.savgol_filter(
                    x, window_length=SMOOTHING_WINDOW, polyorder=POLY_ORDER
                )
            except ValueError:
                x_smooth = x  # Fallback

            # Derivatives (Kinematics)
            vel = np.gradient(x_smooth)
            acc = np.gradient(vel)

            # Statistical Descriptors for Smooth, Velocity, Acceleration
            for name, sig in [("smooth", x_smooth), ("vel", vel), ("acc", acc)]:
                features[f"{col}_{name}_mean"] = np.mean(sig)
                features[f"{col}_{name}_std"] = np.std(sig)
                features[f"{col}_{name}_q01"] = np.quantile(sig, 0.01)
                features[f"{col}_{name}_q99"] = np.quantile(sig, 0.99)

            # --- View 3: Structural Spectral Features ---
            # Welch's method for Power Spectral Density
            f, Pxx = signal.welch(x_smooth, nperseg=256)
            features[f"{col}_spec_mean"] = np.mean(Pxx)
            features[f"{col}_spec_std"] = np.std(Pxx)
            features[f"{col}_spec_max"] = np.max(Pxx)

            # Spectral Centroid
            sum_Pxx = np.sum(Pxx)
            if sum_Pxx == 0:
                features[f"{col}_spec_centroid"] = 0
            else:
                features[f"{col}_spec_centroid"] = np.sum(f * Pxx) / sum_Pxx

            # --- View 4: Flattened Temporal Windows ---
            n_windows = 10
            wins = np.array_split(x, n_windows)
            for i, w in enumerate(wins):
                features[f"{col}_win_{i}_rms"] = np.sqrt(np.mean(w**2))
                features[f"{col}_win_{i}_mean"] = np.mean(w)

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_dataset(metadata_df, dataset_name, load_cached=True):
    """
    Process a dataset (train/val/test) with caching mechanism.
    """
    cache_file = os.path.join(WORKING_DIR, f"{dataset_name}_features.parquet")

    if load_cached and os.path.exists(cache_file):
        print(f"Loading cached features for {dataset_name} from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Generating features for {dataset_name}...")

    # Parallel processing for efficiency
    results = joblib.Parallel(n_jobs=N_JOBS)(
        joblib.delayed(extract_segment_features)(row["file_path"], row["segment_id"])
        for _, row in metadata_df.iterrows()
    )

    # Filter out any failed files
    results = [r for r in results if r is not None]

    features_df = pd.DataFrame(results)

    # Save to cache
    print(f"Saving features for {dataset_name} to {cache_file}...")
    features_df.to_parquet(cache_file)

    return features_df


# ==========================================
# MAIN PIPELINE
# ==========================================


def run_pipeline():
    print("Starting Peak-Aware Stacked Kinematic Ensemble Pipeline...")

    # 1. Load Metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 2. Feature Engineering
    train_features = process_dataset(train_meta, "train")
    val_features = process_dataset(val_meta, "val")
    test_features = process_dataset(test_meta, "test")

    # Merge targets
    train_df = train_features.merge(
        train_meta[["segment_id", "time_to_eruption"]], on="segment_id"
    )
    val_df = val_features.merge(
        val_meta[["segment_id", "time_to_eruption"]], on="segment_id"
    )

    # Combine Train and Val for robust Stacking
    feature_cols = [
        c for c in train_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]

    X_train_full = pd.concat(
        [train_df[feature_cols], val_df[feature_cols]], axis=0
    ).reset_index(drop=True)
    y_train_full = pd.concat(
        [train_df["time_to_eruption"], val_df["time_to_eruption"]], axis=0
    ).reset_index(drop=True)
    X_test = test_features[feature_cols]

    print(f"Combined Training Data Shape: {X_train_full.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # 3. Model Training (Stacking)

    # --- Level 0 Models ---
    lgbm_model = lgb.LGBMRegressor(
        objective="regression",
        metric="mae",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_SEED,
        verbosity=-1,
        n_jobs=N_JOBS,
    )

    xgb_model = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        tree_method="hist",
    )

    cat_model = CatBoostRegressor(
        loss_function="MAE",
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_SEED,
        verbose=0,
        allow_writing_files=False,
    )

    models = {"lgbm": lgbm_model, "xgb": xgb_model, "cat": cat_model}

    # Stacking Arrays
    oof_preds = pd.DataFrame(index=X_train_full.index)
    test_preds_level0 = {k: np.zeros(len(X_test)) for k in models.keys()}

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    print("\n--- Level 0 Training (Base Learners) ---")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_full)):
        X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        for name, model in models.items():
            # Train with Early Stopping
            if name == "cat":
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False,
                )
            elif name == "lgbm":
                callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
            elif name == "xgb":
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            # Predict
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

            # Store OOF
            if fold == 0:
                oof_preds[name] = np.nan
            oof_preds.loc[val_idx, name] = val_pred

            # Accumulate Test Preds (Average across folds)
            test_preds_level0[name] += test_pred / N_FOLDS

            score = mean_absolute_error(y_val, val_pred)
            print(f"Fold {fold+1} - {name} MAE: {score:.4f}")

    # --- Level 1 Training (Meta Learner) ---
    print("\n--- Level 1 Training (Ridge Stacking) ---")
    meta_model = Ridge(alpha=1.0, random_state=RANDOM_SEED)

    # Features for meta model are the predictions from Level 0
    X_meta_train = oof_preds[list(models.keys())]
    X_meta_test = pd.DataFrame(test_preds_level0)

    meta_model.fit(X_meta_train, y_train_full)

    # Evaluate Meta Model on OOF (Approximation)
    meta_oof_pred = meta_model.predict(X_meta_train)
    meta_score = mean_absolute_error(y_train_full, meta_oof_pred)
    print(f"Meta-Learner OOF MAE: {meta_score:.4f}")

    # Final Prediction
    final_preds = meta_model.predict(X_meta_test)

    # 4. Submission
    submission = pd.DataFrame(
        {"segment_id": test_features["segment_id"], "time_to_eruption": final_preds}
    )

    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Pipeline Completed Successfully.")


# Execute Pipeline
if __name__ == "__main__":
    run_pipeline()
