import os
import sys
import numpy as np
import pandas as pd
import torch
import gc

# Import from provided libraries
from library.config import Config
from library.utils import setup_logger, compute_mcc
from library.trainer import Trainer
from library.data_manager import DataManager


def main():
    # 1. Setup Logger
    logger = setup_logger("runfile")
    logger.info("Starting Fast Baseline Run...")

    # 2. Runtime Configuration Patching (Fast Baseline)
    # Reduce epochs to ensure execution finishes well within the time limit
    logger.info("Patching Config for Fast Baseline execution...")
    Config.TRAINING["SCOUT_EPOCHS"] = 100
    Config.TRAINING["EXPERT_EPOCHS"] = 300

    # Ensure GPU is used if available (though Tree models in Config default to CPU n_jobs)
    # We respect the provided Config for model parameters to ensure stability,
    # but we verify environment capabilities.
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("No GPU detected, running on CPU.")

    # 3. Initialize Components
    data_manager = DataManager()
    trainer = Trainer()

    # 4. Load Data
    # We use load_cached=True to leverage any pre-computed features in ./working
    logger.info("Loading Datasets...")
    df_train = data_manager.load_train_features(load_cached=True)
    df_val = data_manager.load_val_features(load_cached=True)
    df_test = data_manager.load_test_features(load_cached=True)

    # Get Validation Arrays
    X_val, y_val = data_manager.get_validation_set(df_val)

    # 5. Phase 1: Train Scouts
    # Train lightweight models to find hard negatives
    scout_lgbm, scout_xgb = trainer.train_scouts(df_train, X_val, y_val)

    # Force garbage collection to manage memory
    gc.collect()

    # 6. Phase 2: Mine Hard Negatives
    # Identify samples where scouts fail
    hard_neg_indices = trainer.mine_hard_negatives(
        df_train, scout_lgbm, scout_xgb, load_cached=True
    )

    # Free up scout memory
    del scout_lgbm, scout_xgb
    gc.collect()

    # 7. Phase 3: Train Expert Ensemble
    # Train the final robust ensemble on enriched data
    ensemble = trainer.train_expert_ensemble(df_train, hard_neg_indices, X_val, y_val)

    # 8. Validation & Threshold Optimization
    logger.info("Performing Validation Inference...")

    # Predict on validation set
    val_probs = ensemble.predict(X_val)

    # Find best threshold
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_mcc = -1.0
    best_thresh = 0.5

    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        score = compute_mcc(y_val, preds)
        if score > best_mcc:
            best_mcc = score
            best_thresh = t

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {best_mcc}")
    logger.info(f"Best Threshold: {best_thresh}")

    # 9. Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate absolute error magnitude
    errors = np.abs(y_val - val_probs)

    # Create a DataFrame for correlation analysis
    df_analysis = pd.DataFrame(X_val, columns=data_manager.feature_cols)
    df_analysis["error_magnitude"] = errors

    # Compute correlation
    correlations = df_analysis.corr()["error_magnitude"].sort_values(ascending=False)

    print("\nFailure Analysis - Feature Correlations with Error:")
    print(correlations.head(5))
    print("-" * 30)

    # 10. Submission Generation
    # Only submit if metric meets the requirement
    SUBMISSION_THRESHOLD = 0.6865

    if best_mcc > SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation MCC ({best_mcc}) > {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        trainer.generate_submission(ensemble, best_thresh, df_test)
    else:
        logger.warning(
            f"Validation MCC ({best_mcc}) <= {SUBMISSION_THRESHOLD}. Skipping submission."
        )

    logger.info("Runfile execution complete.")


if __name__ == "__main__":
    main()
