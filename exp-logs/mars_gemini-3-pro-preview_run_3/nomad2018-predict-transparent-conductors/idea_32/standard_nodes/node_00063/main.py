import os
import sys
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

# Import from provided libraries
from library.data_pipeline import DatasetLoader
from library.training import train_and_evaluate
from library.config import TARGET_COLS, XGB_PARAMS, RANDOM_SEED, set_seed


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    print("Initializing pipeline...")

    # 2. Load Data
    # The loader handles feature generation and caching
    loader = DatasetLoader()
    train_df, val_df, test_df = loader.load_all_data(
        load_cached_data=True, clean_data=True
    )

    # 3. Prepare Feature Matrices
    # Columns to exclude from features
    exclude_cols = ["id", "file_path"] + TARGET_COLS

    # Identify feature columns (intersection of columns in all dfs to be safe)
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Ensure test_df has the same feature columns (it shouldn't have targets)
    # Align columns just in case cleaning dropped different ones, though loader handles this together usually.
    # We use the columns present in train_df.
    feature_cols = [c for c in feature_cols if c in test_df.columns]

    print(f"Training with {len(feature_cols)} features.")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLS]

    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COLS]

    X_test = test_df[feature_cols]
    test_ids = test_df["id"]

    # 4. Configure Model (GPU Optimization)
    current_params = XGB_PARAMS.copy()
    if torch.cuda.is_available():
        print("GPU detected. Configuring XGBoost to use CUDA.")
        current_params["device"] = "cuda"
        current_params["tree_method"] = "hist"
    else:
        print("No GPU detected. Using CPU.")

    # 5. Train and Evaluate
    # We use a reasonable number of estimators for a baseline, early stopping will handle the rest.
    # The config has 3000, which is fine with early stopping.
    model, metrics = train_and_evaluate(
        X_train, y_train, X_val, y_val, params=current_params
    )

    # 6. Validation Assessment
    # The metric defined is Column-wise root mean squared logarithmic error.
    # Our training module calculates RMSLE for each column.
    # The competition metric is usually the mean of these RMSLEs.

    rmsle_formation = metrics[f"rmsle_{TARGET_COLS[0]}"]
    rmsle_bandgap = metrics[f"rmsle_{TARGET_COLS[1]}"]
    final_metric = (rmsle_formation + rmsle_bandgap) / 2.0

    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Generate predictions on validation set again to analyze errors (model.predict handles inverse transform)
    val_preds = model.predict(X_val)

    # Calculate log-space errors (since metric is RMSLE)
    # Error magnitude = |log1p(pred) - log1p(true)|
    error_df = pd.DataFrame(index=X_val.index)

    for target in TARGET_COLS:
        # Avoid log(negative) issues by clipping, though predictions should be non-negative
        p = np.maximum(val_preds[target], 0)
        t = np.maximum(y_val[target], 0)
        error = np.abs(np.log1p(p) - np.log1p(t))
        error_df[f"error_{target}"] = error

    # Correlate errors with features
    # We combine features and errors
    analysis_df = pd.concat([X_val, error_df], axis=1)

    # Calculate correlations
    correlations = analysis_df.corr()

    for target in TARGET_COLS:
        err_col = f"error_{target}"
        if err_col in correlations.columns:
            print(f"\nTop feature correlations with error in {target}:")
            # Sort by absolute correlation
            corr_series = correlations[err_col].drop(
                error_df.columns, errors="ignore"
            )  # Drop error cols
            top_corr = corr_series.abs().sort_values(ascending=False).head(5)
            for feat, val in top_corr.items():
                sign = correlations.loc[feat, err_col]
                print(f"  {feat}: {sign:.4f}")

    # 8. Submission
    threshold = 0.05095
    if final_metric < threshold:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({threshold}). Generating submission..."
        )

        # Predict on test set
        test_preds = model.predict(X_test)

        # Construct submission DataFrame
        submission_df = pd.DataFrame(
            {
                "id": test_ids,
                "formation_energy_ev_natom": test_preds["formation_energy_ev_natom"],
                "bandgap_energy_ev": test_preds["bandgap_energy_ev"],
            }
        )

        # Save
        save_path = os.path.join(submission_dir, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nValidation metric ({final_metric}) does NOT meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
