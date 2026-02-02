import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import workflow functions and configuration
from library.workflow import train_and_evaluate, generate_submission
from library.feature_engineering import FeaturePipeline
from library.config import TARGET_COLS, RANDOM_SEED


def main():
    # 1. Train the model using the workflow
    # We use the full dataset (sample_size=None) as the dataset size is manageable (approx 2000 samples)
    # and we want the best possible performance.
    print("Starting training pipeline...")
    predictor = train_and_evaluate(sample_size=None, load_cached_data=True)

    # 2. Load Validation Data for Assessment and Failure Analysis
    # We re-load the validation set here to perform explicit metric calculation and failure analysis
    print("\nLoading validation data for analysis...")
    pipeline = FeaturePipeline()
    val_df = pipeline.process_split(
        split="val", sample_size=None, load_cached_data=True
    )

    # 3. Compute Final Validation Metric
    print("\nComputing final validation metrics...")
    val_preds = predictor.predict(val_df)

    rmsle_scores = []
    for target in TARGET_COLS:
        y_true = val_df[target]
        y_pred = val_preds[target]
        # RMSLE: sqrt(mean((log1p(true) - log1p(pred))^2))
        score = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
        rmsle_scores.append(score)
        print(f"RMSLE for {target}: {score}")

    # The metric is Column-wise root mean squared logarithmic error.
    # Usually this implies the mean of the RMSLEs of the columns.
    final_metric = np.mean(rmsle_scores)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude per sample (mean squared log error across targets)
    # We use this to find which features correlate with high error
    error_per_sample = np.zeros(len(val_df))
    for target in TARGET_COLS:
        y_true = val_df[target]
        y_pred = val_preds[target]
        error_per_sample += (np.log1p(y_true) - np.log1p(y_pred)) ** 2

    # Take sqrt and average to get a per-sample RMSLE-like magnitude
    error_magnitude = np.sqrt(error_per_sample / len(TARGET_COLS))

    # Prepare dataframe for correlation
    analysis_df = val_df.select_dtypes(include=[np.number]).copy()
    # Drop target columns and id from analysis if present
    cols_to_drop = TARGET_COLS + ["id"]
    analysis_df = analysis_df.drop(
        columns=[c for c in cols_to_drop if c in analysis_df.columns], errors="ignore"
    )

    # Add error magnitude
    analysis_df["error_magnitude"] = error_magnitude

    # Calculate correlations
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

    # Sort by absolute correlation
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print("Top feature correlations with error magnitude:")
    print(top_correlations)

    # Print sign of correlation for top features
    print("\nDirection of top correlations:")
    for feature in top_correlations.index:
        corr_value = correlations[feature]
        print(f"{feature}: {corr_value:.4f}")

    # 5. Generate Submission
    # Threshold check
    THRESHOLD = 0.057877
    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(predictor, sample_size=None, load_cached_data=True)
    else:
        print(
            f"\nValidation metric {final_metric} is NOT better than threshold {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
