import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Ensure the library modules can be imported from the current directory
sys.path.append(os.getcwd())

from library.config import Config
from library.features import process_data
from library.pipeline import run_training_pipeline, run_inference_pipeline


def main():
    # 1. Execute Training Pipeline
    # This handles data loading, feature extraction (with caching), model training, and basic evaluation.
    print("--- Executing Training Pipeline ---")
    artifacts = run_training_pipeline(load_cached_data=True)

    # 2. Validation & Failure Analysis
    print("\n--- Performing Validation & Failure Analysis ---")

    # Load validation data from cache (computed during training pipeline)
    df_val = process_data(split="val", load_cached_data=True)

    # Retrieve trained artifacts
    models = artifacts["models"]
    cleaner = artifacts["cleaner"]
    target_transformer = artifacts["target_transformer"]

    # Prepare validation features
    exclude_cols = ["id"] + Config.TARGET_COLS
    feature_cols = [c for c in df_val.columns if c not in exclude_cols]

    # Apply the same feature cleaning/selection as used in training
    X_val = df_val[feature_cols]
    X_val_clean = cleaner.transform(X_val)

    rmsle_scores = []

    for target in Config.TARGET_COLS:
        if target not in models:
            continue

        model = models[target]

        # Get true values and transform to log space (metric space)
        y_true = df_val[target].values
        y_true_log = target_transformer.transform(y_true)

        # Predict (model outputs are already in log space)
        y_pred_log = model.predict(X_val_clean)

        # Calculate RMSLE for this column
        # RMSLE = RMSE(log(1+y), log(1+pred))
        rmsle = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
        rmsle_scores.append(rmsle)

        # Failure Analysis: Calculate error magnitude
        error_magnitude = np.abs(y_true_log - y_pred_log)

        # Calculate correlation between error and features to identify failure modes
        if isinstance(X_val_clean, pd.DataFrame):
            # Compute correlations between error magnitude and feature values
            corrs = X_val_clean.corrwith(
                pd.Series(error_magnitude, index=X_val_clean.index)
            )
            # Identify top 5 features most associated with error
            top_features = corrs.abs().sort_values(ascending=False).head(5)

            print(f"\nTop error-correlated features for {target}:")
            print(top_features)

    # Calculate Final Metric (Mean Column-wise RMSLE)
    final_metric = np.mean(rmsle_scores)

    # Print the metric in the required format
    print(f"\nFinal Validation Metric: {final_metric}")

    # 3. Submission Logic
    # Threshold defined in the task description
    THRESHOLD = 0.05500532306811823

    if final_metric < THRESHOLD:
        print(f"\nValidation metric {final_metric} meets threshold {THRESHOLD}.")
        print("Proceeding to inference on test set...")
        run_inference_pipeline(artifacts, load_cached_data=True)
    else:
        print(
            f"\nValidation metric {final_metric} does not meet threshold {THRESHOLD}."
        )
        print("Submission generation skipped.")


if __name__ == "__main__":
    main()
