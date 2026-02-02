import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.data_loader import load_data
from library.feature_engineering import process_data
from library.stacking_engine import StackingEngine
from library.utils import setup_logger, save_submission


def main():
    # 1. Initialization and Setup
    set_seed(Config.SEED)
    logger = setup_logger("runfile")
    logger.info("Starting runfile execution...")

    # 2. Load Data
    # Returns DataFrames: df_train, df_val, df_test
    # We use cached data if available to save time
    logger.info("Loading data...")
    df_train, df_val, df_test = load_data(load_cached_data=True)

    # 3. Feature Engineering
    # Returns dictionary of numpy arrays for Linear and Tree branches
    logger.info("Processing features...")
    data = process_data(df_train, df_val, df_test, load_cached_data=True)

    # Extract arrays for clarity
    X_train_lin = data["train_linear"]
    X_train_tree = data["train_tree"]
    y_train = data["y_train"]

    X_val_lin = data["val_linear"]
    X_val_tree = data["val_tree"]
    y_val = data["y_val"]

    X_test_lin = data["test_linear"]
    X_test_tree = data["test_tree"]

    # 4. Model Training (Stacking Ensemble)
    engine = StackingEngine()

    # 4a. Cross-Validation (Level 1)
    # Generates OOF predictions for the training set to train the meta-learner
    logger.info("Running Cross-Validation on Training Set...")
    oof_lin, oof_tree = engine.train_cv(X_train_lin, X_train_tree, y_train)

    # 4b. Train Meta-Learner (Level 2)
    # Uses OOF predictions from base models
    logger.info("Training Meta-Learner...")
    engine.train_meta_learner(oof_lin, oof_tree, y_train)

    # 4c. Retrain Base Models
    # Retrains base models on the full training set for final inference
    logger.info("Retraining Base Models on Full Training Set...")
    engine.train_final_base_models(X_train_lin, X_train_tree, y_train)

    # 5. Validation Inference and Evaluation
    logger.info("Evaluating on Validation Set...")
    val_preds = engine.predict(X_val_lin, X_val_tree)

    # Compute Metric
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y_val - val_preds)

    # Create analysis DataFrame
    # We use the original validation DataFrame to access metadata/features
    analysis_df = df_val.copy()
    analysis_df["error"] = errors

    # Identify numerical columns for correlation analysis
    # We exclude the target and the error column itself
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["requester_received_pizza", "error", "sample_index"]
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = {}
    for col in numeric_cols:
        # Simple correlation ignoring NaNs (filled with 0 or dropped)
        # Using fillna(0) for robustness in this quick analysis
        try:
            corr = analysis_df[col].fillna(0).corr(analysis_df["error"])
            if not np.isnan(corr):
                correlations[col] = corr
        except Exception:
            pass

    # Sort by magnitude of correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\n--- Failure Analysis: Correlation of Features with Prediction Error ---")
    for feat, corr in sorted_corrs[:10]:
        print(f"{feat}: {corr:.4f}")
    print("-----------------------------------------------------------------------\n")

    # 7. Submission Generation
    # Threshold defined in task
    TARGET_THRESHOLD = 0.6994047619047619

    if val_auc > TARGET_THRESHOLD:
        logger.info(
            f"Validation AUC ({val_auc}) exceeds threshold ({TARGET_THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_preds = engine.predict(X_test_lin, X_test_tree)

        # Retrieve Request IDs
        request_ids = df_test["request_id"].values

        # Save Submission
        save_submission(request_ids, test_preds, Config.SUBMISSION_FILE)
    else:
        logger.warning(
            f"Validation AUC ({val_auc}) does not exceed threshold ({TARGET_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
