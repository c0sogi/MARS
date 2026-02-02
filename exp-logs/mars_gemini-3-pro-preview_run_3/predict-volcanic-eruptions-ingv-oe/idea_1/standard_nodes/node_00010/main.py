import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

from library.config import (
    set_seed,
    SEED,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
)
from library.dataset import load_dataset
from library.model import EruptionPredictor


def main():
    # 1. Setup
    # Set random seeds for reproducibility
    set_seed(SEED)

    # 2. Data Loading
    # We load the full datasets (debug_size=None) to ensure a high-quality baseline.
    # The feature extraction process for the dataset size (~4000 files total) is
    # sufficiently fast (approx. 5-10 mins) to fit well within the time constraints.
    print("Loading Training Data...")
    X_train, y_train = load_dataset(
        TRAIN_META_PATH, is_train=True, load_cached_data=True, debug_size=None
    )

    print("Loading Validation Data...")
    X_val, y_val = load_dataset(
        VAL_META_PATH, is_train=False, load_cached_data=True, debug_size=None
    )

    # 3. Model Training
    print("Training Model...")
    predictor = EruptionPredictor()
    predictor.fit(X_train, y_train, X_val, y_val)

    # 4. Validation Assessment
    print("Performing Validation Inference...")
    val_preds = predictor.predict(X_val)

    # Calculate and print the required metric
    mae = mean_absolute_error(y_val, val_preds)
    print(f"Final Validation Metric: {mae}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error magnitude
    errors = np.abs(y_val - val_preds)

    # Prepare analysis dataframe (exclude non-feature columns like segment_id)
    analysis_df = X_val.copy()
    if "segment_id" in analysis_df.columns:
        analysis_df = analysis_df.drop(columns=["segment_id"])

    analysis_df["error_magnitude"] = errors

    # Calculate correlation between features and error magnitude
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

    # Identify top features correlated with high error
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print("Top 10 features correlated with prediction error magnitude:")
    for feature, _ in top_correlations.items():
        # Print the actual signed correlation value
        val = correlations[feature]
        print(f"{feature}: {val}")

    # 6. Submission Generation
    BASELINE_MAE = 3398603.6592843872
    if mae < BASELINE_MAE:
        print("\nMetric improved. Loading Test Data...")
        X_test, _ = load_dataset(
            TEST_META_PATH, is_train=False, load_cached_data=True, debug_size=None
        )

        # Verify test data shape
        print(f"Test Data Shape: {X_test.shape}")

        print("Generating Submission Predictions...")
        test_preds = predictor.predict(X_test)

        print("Saving Submission...")
        predictor.create_submission(X_test, test_preds)
    else:
        print(
            f"\nMetric {mae} did not improve baseline {BASELINE_MAE}. Skipping submission."
        )


if __name__ == "__main__":
    main()
