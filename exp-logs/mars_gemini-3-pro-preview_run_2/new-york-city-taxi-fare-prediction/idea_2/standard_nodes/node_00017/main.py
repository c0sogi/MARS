import os
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import random

# Import provided library components
from library.config import TRAIN_CONFIG, XGB_PARAMS, PATH_CONFIG, SEED
from library.model_trainer import XGBTrainer
from library.evaluator import ModelEvaluator
from library.data_processor import TaxiDataProcessor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(evaluator, sample_size=500000):
    """
    Analyzes model failures by correlating error magnitude with features.
    """
    # Load validation data
    val_df = evaluator.processor.get_processed_data("val", load_cached_data=True)

    # Sample for analysis speed if needed
    if len(val_df) > sample_size:
        val_df = val_df.sample(n=sample_size, random_state=SEED)

    target_col = "fare_amount"
    drop_cols = ["key", target_col]

    X_val = val_df.drop(columns=drop_cols)
    y_true = val_df[target_col].values

    # Predict
    raw_preds = evaluator.predict(X_val)
    final_preds = evaluator.post_process(raw_preds)

    # Calculate Error
    errors = np.abs(y_true - final_preds)

    # Create Analysis DataFrame
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = errors

    # Calculate Correlations
    correlations = analysis_df.corrwith(analysis_df["error_magnitude"])

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    # Sort by absolute correlation
    sorted_corr = correlations.abs().sort_values(ascending=False)
    for feature, corr_val in sorted_corr.items():
        if feature != "error_magnitude":
            print(f"{feature}: {correlations[feature]:.4f}")


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Train
    # Use standard trainer to train on the FULL dataset (~44M rows)
    # The A100 GPU can handle this efficiently.
    trainer = XGBTrainer()
    trainer.train(load_cached_data=True)

    # 3. Validation
    evaluator = ModelEvaluator()
    # Calculate metrics on the full validation set
    rmse = evaluator.calculate_metrics()
    print(f"Final Validation Metric: {rmse}")

    # 4. Failure Analysis
    perform_failure_analysis(evaluator)

    # 5. Submission
    # Updated threshold based on previous best run
    THRESHOLD = 4.278504866347902

    if rmse < THRESHOLD:
        evaluator.generate_submission()
    else:
        print(
            f"Validation metric {rmse} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
