import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from library.config import TARGET_COLS, RANDOM_SEED
from library.model_wrapper import DualXGBoostModel
from library.preprocessing import get_preprocessed_dataset
from library.data_manager import MaterialDataset

# Ensure reproducibility
np.random.seed(RANDOM_SEED)


def main():
    print("Initializing Model...")
    # Initialize the wrapper class which handles the two XGBoost models
    model_wrapper = DualXGBoostModel()

    # --- Training ---
    print("\nStarting Training...")
    # Train the models. This handles loading train/val data, fitting the preprocessor,
    # and training the XGBoost regressors with early stopping.
    # We use the default n_estimators from config, relying on early stopping for speed.
    model_wrapper.train(load_cached_data=True)

    # --- Validation & Metric Calculation ---
    print("\nPerforming Validation Assessment...")

    # Reload validation data to perform explicit metric calculation and failure analysis
    # The preprocessor state is already set from the training phase
    X_val = get_preprocessed_dataset(
        "val", model_wrapper.preprocessor, load_cached_data=True
    )

    # Load ground truth targets
    md = MaterialDataset()
    val_meta = md.load_metadata("val")
    y_val_true = val_meta.set_index("id")[TARGET_COLS]

    # Align indices (inner join to be safe)
    val_combined = X_val.join(y_val_true, how="inner")
    X_val_aligned = val_combined.drop(columns=TARGET_COLS)
    y_val_aligned = val_combined[TARGET_COLS]

    rmsle_scores = []
    errors_dict = {}

    for target in TARGET_COLS:
        # Get the specific trained model
        xgb_model = model_wrapper.models[target]

        # Predict (output is log-transformed)
        y_pred_log = xgb_model.predict(X_val_aligned)

        # Ground truth (log-transformed for RMSLE calculation)
        y_true_log = model_wrapper.preprocessor.log_transform(y_val_aligned[target])

        # Calculate RMSLE (RMSE in log space)
        rmsle = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
        rmsle_scores.append(rmsle)

        # Store absolute errors (in log space) for failure analysis
        # Error in log space ~ relative error in original space
        errors_dict[f"error_{target}"] = np.abs(y_true_log - y_pred_log)

    # Compute final metric (Column-wise RMSLE)
    final_metric = np.mean(rmsle_scores)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Combine errors with features to find correlations
    error_df = pd.DataFrame(errors_dict, index=X_val_aligned.index)
    analysis_df = X_val_aligned.join(error_df)

    # Calculate correlation between features and error magnitudes
    # We look at the correlation of features with the error of each target
    for target in TARGET_COLS:
        error_col = f"error_{target}"
        if error_col not in analysis_df.columns:
            continue

        print(f"\nTop feature correlations with error for {target}:")
        correlations = (
            analysis_df.corrwith(analysis_df[error_col])
            .abs()
            .sort_values(ascending=False)
        )
        # Filter out the error columns themselves
        correlations = correlations.drop(
            [c for c in errors_dict.keys() if c in correlations.index]
        )
        print(correlations.head(5).to_string())

    # --- Submission Generation ---
    # Threshold check as per instructions
    THRESHOLD = 0.05095

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        model_wrapper.predict(load_cached_data=True)
    else:
        print(
            f"\nValidation metric {final_metric} does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
