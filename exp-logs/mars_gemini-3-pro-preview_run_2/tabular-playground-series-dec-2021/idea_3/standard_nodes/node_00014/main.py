import os
import sys
import numpy as np
import pandas as pd
import random
import warnings
import gc
from sklearn.metrics import accuracy_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import library modules
from library.config import (
    SEED,
    TARGET_COL,
    ID_COL,
    LGBM_PARAMS,
    XGB_PARAMS,
    FINAL_SUBMISSION_PATH,
)
from library.data_processor import process_data
from library.model_wrappers import LGBMWrapper, XGBWrapper
from library.cross_validator import run_stratified_kfold
from library.ensembler import optimize_weights, blend_predictions, generate_submission


def set_seed(seed=SEED):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Set global seed
    set_seed()

    print("Starting execution of runfile.py...")

    # 1. Data Loading and Processing
    # Loads train and val splits from metadata, merges them for K-Fold CV,
    # and performs feature engineering. Uses caching to optimize runtime.
    print("Loading and processing data...")
    df_train, df_test = process_data(load_cached_data=True)

    # Extract true labels for OOF evaluation
    y_true = df_train[TARGET_COL]

    # 2. Model Training (Stratified K-Fold)

    # --- LightGBM ---
    print("\nTraining LightGBM Ensemble (Stratified K-Fold)...")
    lgbm_oof, lgbm_test, classes = run_stratified_kfold(
        LGBMWrapper, LGBM_PARAMS, df_train, df_test, verbose=True
    )

    # Force garbage collection to free GPU memory
    gc.collect()

    # --- XGBoost ---
    print("\nTraining XGBoost Ensemble (Stratified K-Fold)...")
    xgb_oof, xgb_test, _ = run_stratified_kfold(
        XGBWrapper, XGB_PARAMS, df_train, df_test, verbose=True
    )

    gc.collect()

    # 3. Ensemble Optimization
    print("\nOptimizing Ensemble Weights...")
    oof_preds_dict = {"lgbm": lgbm_oof, "xgb": xgb_oof}

    # Optimize weights based on OOF Log Loss
    weights = optimize_weights(oof_preds_dict, y_true, classes)

    # 4. Final Validation Metric Calculation
    # Compute weighted OOF probabilities
    final_oof_probs = (weights["lgbm"] * lgbm_oof) + (weights["xgb"] * xgb_oof)

    # Convert probabilities to class labels
    final_pred_indices = np.argmax(final_oof_probs, axis=1)
    final_pred_labels = classes[final_pred_indices]

    # Calculate Accuracy
    final_acc = accuracy_score(y_true, final_pred_labels)

    # PRINT REQUIRED METRIC
    # This metric represents the accuracy on the full training set using
    # unbiased Out-Of-Fold predictions (equivalent to a rigorous validation score).
    print(f"Final Validation Metric: {final_acc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate binary error vector (1 for error, 0 for correct)
    errors = (y_true != final_pred_labels).astype(int)

    # Select feature columns (exclude ID and Target)
    feature_cols = [c for c in df_train.columns if c not in [ID_COL, TARGET_COL]]
    features = df_train[feature_cols]

    # Compute correlation between features and error to identify weak sub-domains
    print("Computing feature correlations with error...")
    correlations = features.corrwith(errors)

    # Sort by absolute correlation
    abs_corrs = correlations.abs().sort_values(ascending=False)

    print("Top 10 features correlated with prediction error:")
    print(abs_corrs.head(10))

    # 6. Submission Generation
    THRESHOLD = 0.9614974893048581

    if final_acc > THRESHOLD:
        print(f"\nValidation Metric ({final_acc}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission file...")

        test_preds_dict = {"lgbm": lgbm_test, "xgb": xgb_test}

        # Blend test predictions using the optimized weights
        blended_test_probs = blend_predictions(test_preds_dict, weights)

        # Get Test IDs
        test_ids = df_test[ID_COL]

        # Generate CSV
        generate_submission(
            test_ids, blended_test_probs, classes, FINAL_SUBMISSION_PATH
        )

    else:
        print(
            f"\nValidation Metric ({final_acc}) does not exceed threshold ({THRESHOLD})."
        )
        print("Submission generation skipped.")


if __name__ == "__main__":
    main()
