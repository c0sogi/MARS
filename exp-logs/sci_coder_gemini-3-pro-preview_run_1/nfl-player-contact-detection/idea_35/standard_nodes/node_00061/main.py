import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef

# Import from provided library
from library.config import Config
from library.utils import seed_everything, setup_logger, ensure_dir
from library.training_pipeline import run_training_pipeline, get_feature_cols
from library.inference_pipeline import run_inference
from library.feature_engineering import generate_features
from library.model_factory import get_model

# Initialize Logger
logger = setup_logger("runfile")


def main():
    # 1. Setup and Config Overrides for Fast Baseline
    seed_everything(Config.SEED)

    # Override Config for speed (Fast Baseline requirements)
    # Reducing estimators to ensure completion within 2 hours
    Config.LGBM_PARAMS["n_estimators"] = 300
    Config.XGB_PARAMS["n_estimators"] = 300
    Config.CATBOOST_PARAMS["iterations"] = 300

    # Override Submission Path to match requirements
    Config.SUBMISSION_OUTPUT_PATH = "./submission/submission.csv"
    ensure_dir(Config.SUBMISSION_OUTPUT_PATH)

    logger.info("Configuration updated for fast baseline execution.")
    logger.info(f"Submission path set to: {Config.SUBMISSION_OUTPUT_PATH}")

    # 2. Run Training Pipeline
    # This handles Feature Generation, Scout Training, Hard Negative Mining,
    # Expert Training, and Threshold Optimization.
    # We use load_cached_data=True to leverage any existing work.
    try:
        run_training_pipeline(load_cached_data=True)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

    # 3. Validation Assessment & Failure Analysis
    logger.info("Starting Validation Assessment...")

    # Load Validation Data
    df_val = generate_features("val", load_cached_data=True)
    feature_cols = get_feature_cols(df_val)
    X_val = df_val[feature_cols]
    y_val = df_val["contact"]

    # Load Models
    model_types = ["lgbm", "xgb", "catboost"]
    experts = []
    for m_name in model_types:
        path = os.path.join(Config.MODEL_DIR, f"expert_{m_name}.joblib")
        if os.path.exists(path):
            model = get_model(m_name)
            model.load(path)
            experts.append(model)
        else:
            logger.warning(f"Model {m_name} not found at {path}")

    if not experts:
        logger.error("No expert models found for validation.")
        sys.exit(1)

    # Ensemble Prediction
    val_probs = np.zeros(len(X_val))
    for model in experts:
        val_probs += model.predict_proba(X_val)
    val_probs /= len(experts)

    # Load Threshold
    thresh_path = os.path.join(Config.MODEL_DIR, "best_threshold.npy")
    if os.path.exists(thresh_path):
        threshold = np.load(thresh_path)[0]
    else:
        threshold = 0.5
        logger.warning(f"Threshold file not found. Using default {threshold}")

    # Calculate Metric
    val_preds = (val_probs >= threshold).astype(int)
    mcc = matthews_corrcoef(y_val, val_preds)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {mcc}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate Error Magnitude (Residuals)
    # We use absolute difference between probability and true label
    errors = np.abs(y_val - val_probs)

    # Calculate correlation between features and error
    # We create a temporary DF for correlation calculation
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = errors

    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print("\n--- Failure Analysis: Top Feature Correlations with Error ---")
    print(top_correlations)
    print("-------------------------------------------------------------\n")

    # 4. Submission Generation
    # Requirement: Generate submission if metric > 0.6865
    TARGET_METRIC = 0.6865

    if mcc > TARGET_METRIC:
        logger.info(
            f"Validation Metric ({mcc}) exceeds target ({TARGET_METRIC}). Generating submission..."
        )
        try:
            run_inference(load_cached_data=True)
            logger.info("Submission generation complete.")
        except Exception as e:
            logger.error(f"Inference pipeline failed: {e}")
            sys.exit(1)
    else:
        logger.warning(
            f"Validation Metric ({mcc}) did not meet target ({TARGET_METRIC}). Skipping submission."
        )


if __name__ == "__main__":
    main()
