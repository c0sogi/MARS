import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef

# Import library modules
import library.config as config
from library.utils import seed_everything, setup_logging
from library.trainer import MiningTrainer
from library.inference import InferenceManager
from library.data_factory import DataFactory


def main():
    # 1. Configuration and Setup
    # Override config for fast baseline execution to ensure completion within time limits
    config.N_ESTIMATORS = 500

    # Setup logging and seeds
    logger = setup_logging(log_filename="runfile.log")
    seed_everything(config.SEED)

    logger.info(
        "Starting Geometrically-Gated Full-Context Mining Ensemble (GGFC-ME) execution..."
    )

    # 2. Training Pipeline
    # Initialize Trainer
    trainer = MiningTrainer()

    # Execute Training Curriculum
    # This handles:
    # - Geometric Gating (via DataFactory)
    # - Feature Engineering (via DataFactory)
    # - Scout Model Training
    # - Hard Negative Mining
    # - Expert Ensemble Training
    # - Threshold Optimization
    expert_ensemble, best_threshold = trainer.run()

    logger.info(f"Training Complete. Best Threshold: {best_threshold}")

    # 3. Validation & Metrics
    logger.info("Starting Final Validation...")

    # Load Validation Data (Full, Non-Gated)
    # We use load_cached=True to leverage features computed during the optimization phase if available
    val_factory = DataFactory(mode="val")
    df_val = val_factory.get_val_dataset(load_cached=True)

    # Define columns to ignore (must match Trainer/Inference logic)
    ignore_cols = [
        "contact_id",
        "contact",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "datetime",
        "video_path_endzone",
        "video_path_sideline",
        "video_path_all29",
        "p2_int",
        "step_join",
        "step_temp",
    ]

    # Prepare Features and Target
    y_val = df_val["contact"].values
    feature_cols = [c for c in df_val.columns if c not in ignore_cols]
    X_val = df_val[feature_cols]

    # Inference on Validation Set
    # Note: ensemble.predict returns unweighted average of LGBM and XGB probabilities
    probs_val = expert_ensemble.predict(X_val)
    preds_val = (probs_val >= best_threshold).astype(int)

    # Calculate MCC
    mcc = matthews_corrcoef(y_val, preds_val)

    # Print Required Metric Format
    print(f"Final Validation Metric: {mcc}")

    # 4. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Calculate Error Magnitude (Absolute difference between truth and probability)
    error_magnitude = np.abs(y_val - probs_val)

    # Calculate Correlation with Input Features
    # We focus on numeric features to identify systematic errors
    numeric_features = X_val.select_dtypes(include=[np.number])

    correlations = {}
    for col in numeric_features.columns:
        # Handle potential NaNs in features by filling with 0 (consistent with DataFactory)
        feat_values = numeric_features[col].fillna(0).values

        # Avoid constant columns to prevent division by zero in correlation
        if np.std(feat_values) > 1e-9:
            corr = np.corrcoef(feat_values, error_magnitude)[0, 1]
            correlations[col] = corr
        else:
            correlations[col] = 0.0

    # Sort by absolute correlation to find most impactful features on error
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 10 Feature Correlations with Error Magnitude:")
    for feature, corr in sorted_corrs[:10]:
        print(f"{feature}: {corr:.6f}")

    # 5. Submission Generation
    TARGET_METRIC = 0.6746827603428585

    if mcc > TARGET_METRIC:
        logger.info(
            f"Validation Metric ({mcc}) exceeds target ({TARGET_METRIC}). Generating Submission..."
        )

        # Initialize Inference Manager
        inference_manager = InferenceManager()

        # Generate Submission
        # This loads test data, features, model, and threshold, then saves to ./submission/submission.csv
        inference_manager.generate_submission()

    else:
        logger.warning(
            f"Validation Metric ({mcc}) did not exceed target ({TARGET_METRIC}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
