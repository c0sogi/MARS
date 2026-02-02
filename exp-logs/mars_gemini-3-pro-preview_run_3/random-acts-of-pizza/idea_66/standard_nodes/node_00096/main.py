import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Configuration Monkey-Patching for Speed
# -----------------------------------------------------------------------------
from library.config import Config

# Reduce estimators for fast baseline execution to ensure < 42 min runtime
Config.LEXICAL_BAGGER_PARAMS["n_estimators"] = 50
Config.COMMUNITY_BAGGER_PARAMS["n_estimators"] = 50
Config.SEMANTIC_BOOSTER_PARAMS["n_estimators"] = 100
Config.SEMANTIC_GRADIENT_PARAMS["n_estimators"] = 100
Config.SEMANTIC_BAGGER_PARAMS["n_estimators"] = 50
Config.TEMPORAL_BOOSTER_PARAMS["n_estimators"] = 100
Config.LEXICAL_BAGGER_PARAMS["n_jobs"] = 12
Config.COMMUNITY_BAGGER_PARAMS["n_jobs"] = 12
Config.SEMANTIC_BOOSTER_PARAMS["n_jobs"] = 12
Config.SEMANTIC_GRADIENT_PARAMS["n_jobs"] = 12

# -----------------------------------------------------------------------------
# 2. Import Library Modules
# -----------------------------------------------------------------------------
from library.pipeline import (
    run_training_pipeline,
    run_inference_pipeline,
    _get_feature_set_for_model,
    _is_volatile_model,
)
from library.data_loader import load_dataset, _process_dataframe
from library.feature_engineering import FeatureFactory
from library.model_factory import ModelFactory
from library.trainer import Trainer

# -----------------------------------------------------------------------------
# 3. Main Execution
# -----------------------------------------------------------------------------


def main():
    # Set seeds for reproducibility
    np.random.seed(Config.RANDOM_SEED)

    print("=== Starting Runfile ===")

    # -------------------------------------------------------------------------
    # Step 1: Training
    # -------------------------------------------------------------------------
    print("\n[Step 1] Running Training Pipeline...")
    # We use load_cached_data=False to ensure we start fresh with our patched config
    run_training_pipeline(load_cached_data=False)

    # -------------------------------------------------------------------------
    # Step 2: Validation on Hold-Out Set
    # -------------------------------------------------------------------------
    print("\n[Step 2] Performing Validation on Hold-Out Set (via OOF)...")

    # Load OOF predictions instead of manual prediction to avoid leakage
    oof_path = os.path.join(Config.CACHE_DIR, "oof_predictions.csv")
    if not os.path.exists(oof_path):
        raise FileNotFoundError(f"OOF predictions not found at {oof_path}")

    oof_df = pd.read_csv(oof_path)

    # Determine validation slice
    # Load raw validation data to get the count
    if not os.path.exists(Config.VAL_DATA_PATH):
        raise FileNotFoundError(f"Val data not found at {Config.VAL_DATA_PATH}")
    raw_val = pd.read_parquet(Config.VAL_DATA_PATH)
    n_val = len(raw_val)

    # The dataset was constructed as [Train, Val], so Val is at the end
    val_subset = oof_df.iloc[-n_val:]

    y_val = val_subset["y_true"].values
    val_final_preds = val_subset["y_pred"].values

    # D. Metric
    final_metric = roc_auc_score(y_val, val_final_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Load val_df for failure analysis (metadata)
    val_df = _process_dataframe(raw_val, is_test=False)

    # -------------------------------------------------------------------------
    # Step 3: Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 3] Failure Analysis...")
    # Calculate error magnitude
    errors = np.abs(y_val - val_final_preds)

    # Correlate with metadata features
    analysis_cols = [c for c in Config.METADATA_COLS if c in val_df.columns]
    print("Correlation between Error and Features:")
    for col in analysis_cols:
        if val_df[col].nunique() > 1:
            corr, _ = pearsonr(errors, val_df[col])
            print(f"  {col}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # Step 4: Submission
    # -------------------------------------------------------------------------
    threshold = 0.65
    if final_metric > threshold:
        print(
            f"\n[Step 4] Metric ({final_metric}) > Threshold ({threshold}). Generating Submission..."
        )
        # CRITICAL: We must run inference pipeline with load_cached_data=False
        # because we overwrote the 'X_test' cache with validation data in Step 2.
        # This forces the FeatureFactory to re-generate features for the ACTUAL test set.
        run_inference_pipeline(load_cached_data=False)
    else:
        print(
            f"\n[Step 4] Metric ({final_metric}) <= Threshold ({threshold}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
