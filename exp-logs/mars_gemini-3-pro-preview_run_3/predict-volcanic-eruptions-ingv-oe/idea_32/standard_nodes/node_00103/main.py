import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import library.config as config
import library.feature_engineering as fe
import library.training_manager as tm

# ==========================================
# Configuration
# ==========================================
# Set random seeds for reproducibility
np.random.seed(config.SEED)

# Training data limit to ensure fast execution.
# Using full dataset for optimal performance.
TRAIN_DEBUG_SIZE = None
# Test data must be fully processed for valid submission
TEST_DEBUG_SIZE = None


def main():
    # ==========================================
    # 1. Training & Validation
    # ==========================================
    print("Starting Training Pipeline...")

    # Run Cross-Validation
    # This handles feature generation, training 5 folds, saving models, and computing OOF MAE.
    val_metric = tm.run_cross_validation(
        load_cached_data=True, debug_size=TRAIN_DEBUG_SIZE
    )

    # Print required metric format
    print(f"Final Validation Metric: {val_metric}")

    # ==========================================
    # 2. Failure Analysis
    # ==========================================
    print("\nStarting Failure Analysis on Validation Set...")

    # Reload data to reconstruct OOF predictions for analysis
    # We use the same debug_size to ensure we analyze the exact data used for training
    df_train = fe.get_train_data(load_cached_data=True, debug_size=TRAIN_DEBUG_SIZE)
    df_val = fe.get_val_data(load_cached_data=True, debug_size=TRAIN_DEBUG_SIZE)

    # Identify Validation IDs to filter later (Strict Hold-out Analysis)
    val_ids = set(df_val["segment_id"].values)

    # Combine for CV reconstruction (matching training_manager logic)
    df_full = pd.concat([df_train, df_val], axis=0, ignore_index=True)

    # Prepare Target and Features
    target_col = "time_to_eruption"
    exclude_cols = ["segment_id", target_col]
    feature_cols = [c for c in df_full.columns if c not in exclude_cols]

    X = df_full[feature_cols]
    y = df_full[target_col].values

    # Reconstruct Stratified Splits
    # We must replicate the binning logic used in training_manager to get identical splits
    num_bins = 10
    if len(y) < num_bins:
        num_bins = 2
    try:
        y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")
    except ValueError:
        y_bins = np.zeros(len(y))

    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Container for OOF predictions
    oof_preds = np.zeros(len(df_full))

    # Iterate folds and predict using saved models
    for fold, (_, val_idx) in enumerate(skf.split(X, y_bins)):
        model_path = os.path.join(config.WORKING_DIR, f"lgbm_fold_{fold}.txt")

        if os.path.exists(model_path):
            # Load model
            model = lgb.Booster(model_file=model_path)

            # Predict on validation chunk
            X_val_fold = X.iloc[val_idx]
            # Note: Saved models are already stripped to best_iteration, so we just predict
            preds = model.predict(X_val_fold)
            oof_preds[val_idx] = preds

    # Calculate Errors
    errors = np.abs(y - oof_preds)

    # Filter for the specific Validation Set (metadata/val.csv)
    # This ensures we are analyzing the "hold-out" performance specifically
    val_mask = df_full["segment_id"].isin(val_ids)

    if val_mask.sum() > 0:
        val_errors = errors[val_mask]
        val_features = X[val_mask]

        print(f"Analyzing {len(val_errors)} validation samples...")

        # Calculate Correlations
        corrs = {}
        for col in feature_cols:
            # Handle potential constant columns or NaNs
            feat_vals = val_features[col].values
            if np.std(feat_vals) > 1e-9:
                c = np.corrcoef(feat_vals, val_errors)[0, 1]
                if not np.isnan(c):
                    corrs[col] = c

        # Sort by absolute correlation
        sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)

        print("\nTop 5 Features Correlated with Error Magnitude:")
        for feat, corr in sorted_corrs[:5]:
            print(f"{feat}: {corr:.4f}")

        print("\n(Positive correlation means higher feature values -> larger errors)")
    else:
        print("Warning: Could not isolate validation set for failure analysis.")

    # ==========================================
    # 3. Submission
    # ==========================================
    THRESHOLD = 2617304.0647319085

    print("\nChecking Submission Criteria...")
    if val_metric < THRESHOLD:
        print(f"Metric {val_metric} < {THRESHOLD}. Proceeding to submission.")
        tm.generate_test_predictions(load_cached_data=True, debug_size=TEST_DEBUG_SIZE)
    else:
        print(f"Metric {val_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
