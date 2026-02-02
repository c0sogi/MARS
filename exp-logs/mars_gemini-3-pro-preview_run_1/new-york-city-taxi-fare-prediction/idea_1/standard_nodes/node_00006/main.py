import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import warnings
import os

# Import from the provided library
import library.config as config
import library.utils as utils
from library.data_processor import TaxiDataProcessor
from library.model import FarePredictor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    utils.set_seed(42)

    # 2. Data Processing
    processor = TaxiDataProcessor()

    # Process Training Data
    # Using full dataset for maximum performance
    print("Processing training data...")
    train_df = processor.process_data(
        "train", load_cached_data=True, debug_sample_size=None
    )

    # Process Validation Data
    # We use the full validation set for accurate metric calculation
    print("Processing validation data...")
    val_df = processor.process_data("val", load_cached_data=True)

    # Process Test Data
    print("Processing test data...")
    test_df = processor.process_data("test", load_cached_data=True)

    # 3. Model Training
    print("Initializing and training model...")
    predictor = FarePredictor()
    predictor.fit(train_df, val_df)

    # 4. Evaluation
    print("Evaluating on validation set...")
    val_preds = predictor.predict(val_df)

    # Calculate RMSE
    rmse = np.sqrt(
        mean_squared_error(val_df[config.FEATURE_CONFIG["target_col"]], val_preds)
    )

    # Print required metric
    print(f"Final Validation Metric: {rmse}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    val_df["error_magnitude"] = np.abs(
        val_df[config.FEATURE_CONFIG["target_col"]] - val_preds
    )

    # Calculate correlations between error magnitude and features
    # We select numeric columns from the features used by the model
    analysis_features = predictor.features

    correlations = {}
    for feature in analysis_features:
        if feature in val_df.columns:
            corr = val_df[feature].corr(val_df["error_magnitude"])
            correlations[feature] = corr

    # Sort and print correlations
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Correlation between Error Magnitude and Input Features:")
    for feature, corr in sorted_corrs:
        print(f"{feature}: {corr:.4f}")

    # 6. Submission
    baseline_rmse = 3.8561551764143713
    if rmse < baseline_rmse:
        print(
            f"\nRMSE {rmse:.5f} improved over baseline {baseline_rmse:.5f}. Generating submission..."
        )
        test_preds = predictor.predict(test_df)

        # Create submission DataFrame
        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_preds})

        # Save submission
        submission_path = config.DATA_PATHS["submission"]
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nRMSE {rmse:.5f} did not improve over baseline {baseline_rmse:.5f}. Skipping submission."
        )


if __name__ == "__main__":
    main()
