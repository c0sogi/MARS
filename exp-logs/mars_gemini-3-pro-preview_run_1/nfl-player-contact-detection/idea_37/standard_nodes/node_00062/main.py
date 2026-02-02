import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

# Import from library
from library.config import Config
from library.training import Trainer
from library.inference import InferenceManager
from library.data_processing import DataProcessor
from library.utils import setup_logger, seed_everything


def main():
    # 1. Configuration & Setup
    # Override Config for fast baseline execution
    Config.N_ESTIMATORS = 800
    Config.LGBM_PARAMS["n_estimators"] = 800
    Config.XGB_PARAMS["n_estimators"] = 800

    # Ensure DEBUG is False to process the entire dataset as required for the final metric
    Config.DEBUG = False

    # Setup logger
    logger = setup_logger("runfile")
    seed_everything(Config.SEED)

    logger.info("Starting runfile execution...")
    logger.info(
        f"Configuration: N_ESTIMATORS={Config.N_ESTIMATORS}, DEBUG={Config.DEBUG}"
    )

    # 2. Training Pipeline
    trainer = Trainer()
    try:
        # Run the full training pipeline (Scouts -> Mining -> Experts -> Threshold)
        # load_cached_data=True allows utilizing pre-computed features if available
        trainer.run(load_cached_data=True)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

    # 3. Validation & Metric Calculation
    logger.info("Starting Validation Phase...")

    # Initialize Processor and Inference Manager
    processor = DataProcessor()
    inference = InferenceManager()

    # Load full validation data
    df_val = processor.get_val_data(load_cached=True, debug=Config.DEBUG)

    # Load trained models and optimized threshold
    try:
        expert_lgbm, expert_xgb = inference.load_models()
        threshold = inference.load_threshold()
    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}")
        sys.exit(1)

    # Prepare validation features
    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
    ]
    feature_cols = [c for c in df_val.columns if c not in meta_cols]

    X_val = df_val[feature_cols]
    y_true = df_val["contact"].values

    # Generate Predictions (Ensemble)
    # Using unweighted average as per Config/Training logic
    p_lgbm = expert_lgbm.predict(X_val)
    p_xgb = expert_xgb.predict(X_val)

    w_lgbm = Config.ENSEMBLE_WEIGHTS["lgbm"]
    w_xgb = Config.ENSEMBLE_WEIGHTS["xgb"]

    y_pred_prob = (p_lgbm * w_lgbm) + (p_xgb * w_xgb)
    y_pred_bin = (y_pred_prob > threshold).astype(int)

    # Calculate Metric
    mcc = matthews_corrcoef(y_true, y_pred_bin)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {mcc:.16f}")

    # 4. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Calculate Error Magnitude
    error_magnitude = np.abs(y_true - y_pred_prob)

    # Calculate correlations
    # We create a temporary DF for correlation calculation
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = error_magnitude

    # Compute correlation of features with error magnitude
    correlations = analysis_df.corrwith(analysis_df["error_magnitude"])

    # Sort by absolute correlation
    correlations_abs = correlations.abs().sort_values(ascending=False)

    print("\nFailure Analysis - Feature Correlations with Error Magnitude:")
    print(correlations_abs.head(10))

    # 5. Submission
    SUBMISSION_THRESHOLD = 0.6865

    if mcc > SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation MCC ({mcc:.4f}) > {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        try:
            # Generate submission using the InferenceManager
            # This handles loading test data, predicting, and saving to CSV
            inference.generate_submission(
                load_cached_data=True, run_validation_optimization=False
            )
        except Exception as e:
            logger.error(f"Submission generation failed: {e}")
    else:
        logger.warning(
            f"Validation MCC ({mcc:.4f}) did not meet threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
