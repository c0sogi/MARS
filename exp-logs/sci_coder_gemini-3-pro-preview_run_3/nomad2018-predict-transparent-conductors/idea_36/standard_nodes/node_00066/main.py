import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import necessary functions and variables from the provided library files
import library.config
from library.config import RANDOM_SEED, TARGET_COLS
from library.utils import log_transform
from library.data_loader import load_metadata
from library.feature_processor import process_dataset
from library.model_handler import train_xgboost, predict


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)
    print("Starting pipeline execution...")

    # 2. Train Models
    # This step handles feature extraction for train/val sets and trains XGBoost models
    # load_cached_data=True allows resuming from checkpoints or using pre-computed features
    models = train_xgboost(load_cached_data=True, debug=False)

    # 3. Validation Assessment
    print("\n" + "=" * 40)
    print("Validation Assessment")
    print("=" * 40)

    # Load validation features (should be cached by the training step)
    val_features_df = process_dataset("val", load_cached_data=True, debug=False)

    # Load validation metadata to get ground truth targets
    val_meta_df = load_metadata("val", debug=False)

    # Merge features with targets
    val_merged = val_features_df.merge(
        val_meta_df[["id"] + TARGET_COLS], on="id", how="inner"
    )

    # Prepare validation feature matrix X_val
    # We must ensure the columns match exactly what the model was trained on
    # Get feature names from one of the trained models
    first_model = models[TARGET_COLS[0]]
    model_feature_names = first_model.get_booster().feature_names

    X_val = val_merged[model_feature_names]

    rmsle_scores = []
    errors_dict = {}

    for target in TARGET_COLS:
        # Get ground truth and apply log transform (since models predict log(1+y))
        y_true_log = log_transform(val_merged[target].values)

        # Predict (model outputs are already in log space)
        y_pred_log = models[target].predict(X_val)

        # Calculate RMSLE (Root Mean Squared Error of Log-transformed values)
        rmsle = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
        rmsle_scores.append(rmsle)

        # Store absolute errors for failure analysis
        errors_dict[target] = np.abs(y_pred_log - y_true_log)

        print(f"Target: {target} | RMSLE: {rmsle}")

    # Compute Final Metric (Column-wise RMSLE, averaged)
    final_metric = np.mean(rmsle_scores)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n" + "=" * 40)
    print("Failure Analysis")
    print("=" * 40)

    # Fill NaNs in features with 0 for correlation calculation
    X_val_filled = X_val.fillna(0)

    for target in TARGET_COLS:
        print(f"\n--- Top Feature Correlations with Error for {target} ---")
        errors = errors_dict[target]
        error_series = pd.Series(errors, index=X_val_filled.index, name="abs_error")

        # Calculate correlation between feature values and error magnitude
        correlations = (
            X_val_filled.corrwith(error_series).abs().sort_values(ascending=False)
        )
        print(correlations.head(5))

    # 5. Submission Generation
    # Threshold check as per requirements
    THRESHOLD = 0.05095

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({THRESHOLD})."
        )
        print("Generating submission file...")
        # The predict function handles test feature extraction, prediction, inverse transform, and saving
        predict(models, load_cached_data=True, debug=False)
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
