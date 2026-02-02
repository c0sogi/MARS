import pandas as pd
import numpy as np
import sys
import os

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.train import run_training
from library.inference import generate_submission
from library.data_loader import load_data
from library.features import extract_features
from library.model import ScoreRegressor
from library.metrics import compute_qwk


def perform_failure_analysis(df_val, y_val, y_pred_continuous):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and essay length.
    """
    print("\n--- Failure Analysis ---")

    # Create a temporary dataframe for analysis
    analysis_df = df_val.copy()
    analysis_df["true_score"] = y_val
    analysis_df["pred_score"] = y_pred_continuous

    # Calculate Error Magnitude (Absolute Error)
    analysis_df["error"] = analysis_df["true_score"] - analysis_df["pred_score"]
    analysis_df["abs_error"] = analysis_df["error"].abs()

    # Calculate Input Feature: Word Count
    # (Using simple split for speed, matching the logic often used in basic EDA)
    analysis_df["word_count"] = analysis_df[Config.TEXT_COL].apply(
        lambda x: len(str(x).split())
    )

    # Calculate Correlation
    correlation = analysis_df["abs_error"].corr(analysis_df["word_count"])

    print(f"Correlation between Absolute Error and Word Count: {correlation:.4f}")

    # Additional Insight: Mean Absolute Error by Score
    print("\nMean Absolute Error (MAE) by True Score:")
    mae_by_score = analysis_df.groupby("true_score")["abs_error"].mean()
    print(mae_by_score)
    print("------------------------\n")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Train the Model
    # This handles loading train/val data, feature extraction, training, and saving artifacts.
    # We use the full dataset (nrows=None) as Ridge Regression is computationally efficient.
    run_training(load_cached_data=True, nrows=None)

    # 3. Validation & Metric Reporting
    # We reload validation data and model to ensure we print the metric in the exact requested format
    # and to perform failure analysis.
    print("Reloading Validation Data for Analysis...")
    df_val = load_data(split="val", load_cached_data=True)

    print("Extracting Validation Features...")
    # extract_features handles loading the vectorizer saved during training
    X_val = extract_features(df_val, split="val")
    y_val = df_val[Config.TARGET_COL].values

    print("Loading Trained Model...")
    model = ScoreRegressor.load(Config.MODEL_PATH)

    print("Generating Validation Predictions...")
    # Get continuous predictions
    y_pred_val_continuous = model.predict(X_val)

    # Round to nearest integer for QWK calculation
    y_pred_val_int = np.round(y_pred_val_continuous).astype(int)

    # Compute Metric
    qwk_score = compute_qwk(y_val, y_pred_val_int)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {qwk_score}")

    # 4. Failure Analysis
    perform_failure_analysis(df_val, y_val, y_pred_val_continuous)

    # 5. Generate Submission
    # This handles loading test data, predicting, and saving the submission file.
    generate_submission(load_cached_data=True)


if __name__ == "__main__":
    main()
