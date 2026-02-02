import sys
import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef

# Import provided library modules
from library.config import Config
from library.utils import calculate_mcc, save_npy
from library.feature_engineering import FeatureEngineer
from library.data_manager import DataManager
from library.model_factory import LGBMWrapper, XGBWrapper
from library.mining_strategy import MiningStrategy


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)

    # Optimize for Fast Baseline execution
    Config.N_ESTIMATORS = 800
    Config.EARLY_STOPPING_ROUNDS = 50

    # Attempt to enable GPU acceleration for Tree Models
    # We update the configuration dictionaries directly
    try:
        import torch

        if torch.cuda.is_available():
            print("GPU detected. Configuring models for GPU acceleration.")
            Config.XGB_PARAMS.update({"device": "cuda", "tree_method": "hist"})
            Config.LGBM_PARAMS.update({"device": "gpu"})
    except ImportError:
        pass

    print(f"Configuration: N_ESTIMATORS={Config.N_ESTIMATORS}")

    # 2. Data Loading & Mining Strategy
    dm = DataManager()
    ms = MiningStrategy()

    print("\n--- Phase 1: Loading Data & Mining ---")
    # Load Training Features (Gated by default in DataManager)
    df_train = dm.get_train_features(load_cached_data=True)

    # Execute Mining Strategy: Train Scouts -> Mine Hard Negatives
    # This ensures we find the edge cases the simple models miss
    hard_indices = ms.execute(df_train, load_cached_data=True)

    # 3. Expert Model Training
    print("\n--- Phase 2: Expert Training ---")
    # Construct Expert Dataset (Positives + Hard Negatives + Buffer)
    X_expert, y_expert = dm.get_expert_dataset(df_train, hard_indices)

    # Clean up full train df to save memory
    del df_train
    gc.collect()

    # Split Expert dataset for internal validation (early stopping)
    X_tr, X_val_int, y_tr, y_val_int = train_test_split(
        X_expert, y_expert, test_size=0.1, random_state=Config.SEED, stratify=y_expert
    )

    # Train Expert LGBM
    print("Training Expert LGBM...")
    expert_lgbm = LGBMWrapper(mode="expert")
    expert_lgbm.fit(X_tr, y_tr, X_val_int, y_val_int)
    expert_lgbm.save(Config.MODEL_EXPERT_LGBM_PATH)

    # Train Expert XGB
    print("Training Expert XGB...")
    expert_xgb = XGBWrapper(mode="expert")
    expert_xgb.fit(X_tr, y_tr, X_val_int, y_val_int)
    expert_xgb.save(Config.MODEL_EXPERT_XGB_PATH)

    # Cleanup
    del X_expert, y_expert, X_tr, X_val_int, y_tr, y_val_int
    gc.collect()

    # 4. Full Validation & Threshold Optimization
    print("\n--- Phase 3: Validation & Optimization ---")
    # We must evaluate on the ENTIRE validation set (including easy negatives)
    # to get a representative MCC. DataManager's default get_val_features applies gating.
    # We manually use FeatureEngineer with apply_gating=False.
    fe = FeatureEngineer()
    print("Generating Full Validation Set (No Gating)...")
    df_val_full = fe._process_dataset(
        Config.VAL_METADATA_PATH, Config.TRAIN_TRACKING_PATH, apply_gating=False
    )

    # Extract features and target
    feature_cols = [c for c in df_val_full.columns if c not in dm.metadata_cols]
    X_val_full = df_val_full[feature_cols]
    y_val_full = df_val_full["contact"].values

    print("Predicting on Validation Set...")
    pred_lgbm = expert_lgbm.predict(X_val_full)
    pred_xgb = expert_xgb.predict(X_val_full)

    # Ensemble Average
    pred_ens = (pred_lgbm + pred_xgb) / 2.0

    # Optimize Threshold
    print("Optimizing Decision Threshold...")
    thresholds = np.linspace(0.1, 0.9, 81)
    best_mcc = -1.0
    best_thresh = 0.5

    for th in thresholds:
        y_pred_bin = (pred_ens > th).astype(int)
        score = calculate_mcc(y_val_full, y_pred_bin)
        if score > best_mcc:
            best_mcc = score
            best_thresh = th

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {best_mcc}")
    print(f"Best Threshold: {best_thresh}")

    # Save threshold
    save_npy(np.array([best_thresh]), Config.BEST_THRESHOLD_PATH)

    # 5. Failure Analysis
    print("\n--- Phase 4: Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(y_val_full - pred_ens)

    # Create a temporary dataframe for correlation analysis
    # Use a subset if data is too large to speed up correlation
    if len(X_val_full) > 100000:
        idx = np.random.choice(len(X_val_full), 100000, replace=False)
        X_analysis = X_val_full.iloc[idx].copy()
        errors_analysis = errors[idx]
    else:
        X_analysis = X_val_full.copy()
        errors_analysis = errors

    X_analysis["error_magnitude"] = errors_analysis

    # Compute correlations
    correlations = (
        X_analysis.corrwith(X_analysis["error_magnitude"])
        .abs()
        .sort_values(ascending=False)
    )

    print("Top Features Correlated with Model Error:")
    print(correlations.head(6))  # Top 5 + error_magnitude itself

    # 6. Submission
    print("\n--- Phase 5: Submission Generation ---")
    SUBMISSION_THRESHOLD = 0.6782

    if best_mcc > SUBMISSION_THRESHOLD:
        print(
            f"Validation Metric ({best_mcc:.4f}) > Threshold ({SUBMISSION_THRESHOLD}). Generating Submission..."
        )

        # Load Test Data (DataManager uses apply_gating=False for test by default)
        df_test = dm.get_test_features(load_cached_data=True)
        X_test = df_test[feature_cols]

        # Predict
        p_lgbm = expert_lgbm.predict(X_test)
        p_xgb = expert_xgb.predict(X_test)
        p_ens = (p_lgbm + p_xgb) / 2.0

        # Apply Threshold
        predictions = (p_ens > best_thresh).astype(int)

        # Create Submission File
        sub_df = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": predictions}
        )

        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation Metric ({best_mcc:.4f}) did not meet threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
