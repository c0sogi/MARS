import os
import sys
import numpy as np
import pandas as pd
import joblib
import gc
import warnings
from sklearn.metrics import matthews_corrcoef

# Set seeds for reproducibility
np.random.seed(42)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# -----------------------------------------------------------------------------
# 1. Configuration & Overrides for Fast Baseline
# -----------------------------------------------------------------------------
# We modify the configuration dictionaries in library.config to speed up training.
# Since dictionaries are mutable, these changes propagate to other modules.
import library.config as config

# Reduce estimators for faster execution
config.LGBM_PARAMS["n_estimators"] = 600
config.XGB_PARAMS["n_estimators"] = 600
config.SCOUT_LGBM_PARAMS["n_estimators"] = 300
config.SCOUT_XGB_PARAMS["n_estimators"] = 300

# Ensure GPU is used where applicable
config.XGB_PARAMS["tree_method"] = "hist"
config.XGB_PARAMS["device"] = "cuda"

# Import library modules after config modification
from library.features import generate_features
from library.trainer import Trainer
from library.models import LGBMClassifierWrapper, XGBClassifierWrapper

# Constants
SAMPLE_SIZE_TRAIN = 200000  # Limit training data for speed
METRIC_THRESHOLD = 0.6865
SUBMISSION_PATH = "./submission/submission.csv"


def main():
    print("Starting SFGA-E Pipeline execution...")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[Step 1] Generating Features...")

    # Load Train Features (Cached if available)
    df_train = generate_features(split="train", load_cached_data=True)

    # Load Val Features (Cached if available)
    # We must use the full validation set for the final metric
    df_val = generate_features(split="val", load_cached_data=True)

    print(f"Original Train Shape: {df_train.shape}")
    print(f"Validation Shape: {df_val.shape}")

    # Downsample Training Data for Fast Baseline
    # We maintain the positive/negative ratio roughly by simple random sampling
    if len(df_train) > SAMPLE_SIZE_TRAIN:
        print(f"Downsampling training data to {SAMPLE_SIZE_TRAIN} rows...")
        df_train_sampled = df_train.sample(
            n=SAMPLE_SIZE_TRAIN, random_state=config.SEED
        ).reset_index(drop=True)
    else:
        df_train_sampled = df_train.copy()

    # -------------------------------------------------------------------------
    # 3. Training Pipeline
    # -------------------------------------------------------------------------
    trainer = Trainer()

    # Phase 1: Train Scouts
    # Scouts are trained on a balanced subset of the sampled training data
    scout_lgbm, scout_xgb = trainer.train_scouts(df_train_sampled)

    # Phase 2: Mine Hard Negatives
    # We mine from the sampled training set to keep it consistent and fast
    hard_neg_indices = trainer.mine_hard_negatives(
        df_train_sampled, scout_lgbm, scout_xgb, load_cache=False
    )

    # Phase 3: Train Experts
    # Experts are trained on Positives + Hard Negatives + Random Anchors
    expert_lgbm, expert_xgb, best_thresh = trainer.train_experts(
        df_train_sampled, hard_neg_indices, df_val
    )

    # Clean up memory
    del df_train, df_train_sampled, scout_lgbm, scout_xgb
    gc.collect()

    # -------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 2] Validation & Failure Analysis...")

    # Prepare Validation Data
    X_val, y_val = trainer._split_X_y(df_val)

    # Predict with Ensemble
    print("Predicting on full validation set...")
    p_val_lgbm = expert_lgbm.predict_proba(X_val)
    p_val_xgb = expert_xgb.predict_proba(X_val)
    p_val_ens = (p_val_lgbm + p_val_xgb) / 2.0

    # Calculate Final Metric
    y_pred_val = (p_val_ens > best_thresh).astype(int)
    final_mcc = matthews_corrcoef(y_val, y_pred_val)

    print(f"Final Validation Metric: {final_mcc}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Error Magnitude (Absolute difference between prob and label)
    # For label 1: error = 1 - p; For label 0: error = p - 0 => |y - p|
    errors = np.abs(y_val - p_val_ens)

    # Correlate errors with features
    # Select numerical features
    numeric_cols = X_val.select_dtypes(include=[np.number]).columns

    # Compute correlations
    # We iterate to avoid creating a massive correlation matrix in memory
    correlations = {}
    for col in numeric_cols:
        try:
            # Simple correlation between feature values and error magnitude
            # Using numpy for speed
            feat_vals = X_val[col].values
            # Handle NaNs if any (though trees handle them, corrcoef doesn't)
            mask = ~np.isnan(feat_vals)
            if np.sum(mask) > 1:
                corr = np.corrcoef(feat_vals[mask], errors[mask])[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr
        except Exception:
            continue

    # Sort and print top correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top 10 Features correlated with Error Magnitude:")
    for name, val in sorted_corr[:10]:
        print(f"{name}: {val:.4f}")

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    if final_mcc > METRIC_THRESHOLD:
        print(
            f"\n[Step 3] Validation Metric ({final_mcc}) > Threshold ({METRIC_THRESHOLD}). Generating Submission..."
        )

        # Load Test Data
        df_test = generate_features(split="test", load_cached_data=True)
        X_test, _ = trainer._split_X_y(df_test)

        # Predict
        print("Predicting on Test Set...")
        p_test_lgbm = expert_lgbm.predict_proba(X_test)
        p_test_xgb = expert_xgb.predict_proba(X_test)
        p_test_ens = (p_test_lgbm + p_test_xgb) / 2.0

        # Apply Threshold
        predictions = (p_test_ens > best_thresh).astype(int)

        # Create Submission DataFrame
        submission = df_test[["contact_id"]].copy()
        submission["contact"] = predictions

        # Save
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
        print(f"Submission shape: {submission.shape}")

    else:
        print(
            f"\n[Step 3] Validation Metric ({final_mcc}) <= Threshold ({METRIC_THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
