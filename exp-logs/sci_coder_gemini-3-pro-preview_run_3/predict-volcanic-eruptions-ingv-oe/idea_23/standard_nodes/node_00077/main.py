import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Import provided library components
from library.config import Config
from library.data_loader import process_dataset
from library.model import run_cross_validation, predict_ensemble


def main():
    # Ensure reproducibility
    np.random.seed(Config.SEED)

    print("=== Starting Pipeline ===")

    # 1. Load Data
    # We load the training metadata and the hold-out validation metadata separately.
    # The feature extraction is handled by process_dataset which uses caching.
    print(f"Loading training data from {Config.TRAIN_META_PATH}...")
    train_df = process_dataset(
        Config.TRAIN_META_PATH, load_cached_data=True, is_test=False
    )

    print(f"Loading validation data from {Config.VAL_META_PATH}...")
    val_df = process_dataset(Config.VAL_META_PATH, load_cached_data=True, is_test=False)

    # 2. Train Ensemble
    # We use the provided run_cross_validation function which trains 5 models
    # using Stratified K-Fold on the provided training dataframe.
    print("\n=== Training Ensemble ===")
    models = run_cross_validation(train_df)

    # 3. Validation Assessment
    print("\n=== Validating on Hold-out Set ===")
    # Prepare validation features
    feature_cols = [
        c for c in val_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    X_val = val_df[feature_cols]
    y_val = val_df["time_to_eruption"]

    # Generate predictions
    val_preds = predict_ensemble(models, X_val)

    # Compute Metric
    final_mae = mean_absolute_error(y_val, val_preds)
    print(f"Final Validation Metric: {final_mae}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute errors
    errors = np.abs(y_val - val_preds)

    # Calculate correlation between features and error magnitude
    # This helps identify which features are associated with hard-to-predict samples
    print("Calculating error correlations...")
    error_correlations = X_val.corrwith(pd.Series(errors, index=X_val.index))

    print("Top 5 features positively correlated with error (Error Drivers):")
    print(error_correlations.sort_values(ascending=False).head(5))

    # 5. Submission Generation
    THRESHOLD = 2617304.0647319085

    if final_mae < THRESHOLD:
        print(
            f"\nMetric ({final_mae}) is below threshold ({THRESHOLD}). Proceeding to submission."
        )

        print(f"Loading test data from {Config.TEST_META_PATH}...")
        test_df = process_dataset(
            Config.TEST_META_PATH, load_cached_data=True, is_test=True
        )

        if test_df.empty:
            print("Error: Test dataframe is empty.")
            return

        print("Generating test predictions...")
        X_test = test_df[feature_cols]
        test_preds = predict_ensemble(models, X_test)

        # Create submission DataFrame
        submission = pd.DataFrame(
            {"segment_id": test_df["segment_id"], "time_to_eruption": test_preds}
        )

        # Ensure submission directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Save submission
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

        # Preview
        print("Submission Head:")
        print(submission.head())

    else:
        print(
            f"\nMetric ({final_mae}) did not pass threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
