import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings

# Import provided library modules
from library import config
from library import data_manager
from library.feature_pipeline import FeatureGenerator
from library.model_handler import TaxiFareRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Configuration and Setup
    set_seed(config.SEED)
    print("Starting Factorized Multi-Moment Hierarchical Gradient Boosting Pipeline...")

    # 2. Data Loading
    # Loads Learner (Train), Wisdom (Priors), Val, and Test sets.
    # Uses caching to speed up execution if files exist.
    print("\n=== Data Loading ===")
    learner_df, wisdom_df, val_df, test_df = data_manager.load_dataset(
        load_cached_data=True
    )

    # 3. Feature Engineering
    # Generates hierarchical spatial moments and applies Conditional Vectorized Subtraction.
    print("\n=== Feature Engineering ===")
    feature_gen = FeatureGenerator()

    # process() returns the feature matrices and targets
    X_train, y_train, X_val, y_val, X_test, test_keys = feature_gen.process(
        learner_df, wisdom_df, val_df, test_df, load_cached_data=True
    )

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # 4. Model Training
    # Initializes XGBoost with GPU support and trains on the enriched data.
    print("\n=== Model Training ===")
    model = TaxiFareRegressor()
    model.train(X_train, y_train, X_val, y_val)

    # 5. Validation Assessment
    print("\n=== Validation Assessment ===")
    # Generate predictions on the full validation set
    # The predict method automatically applies the $2.50 floor
    val_preds = model.predict(X_val)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Final Validation Metric: {rmse}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute errors
    errors = np.abs(y_val - val_preds)

    # Create a DataFrame for correlation analysis
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlation of features with error magnitude
    # We focus on numerical columns
    correlations = analysis_df.corrwith(analysis_df["error_magnitude"]).sort_values(
        ascending=False
    )

    print("Top 5 Features correlated with Error Magnitude:")
    print(correlations.head(6).iloc[1:])  # Skip self-correlation of error_magnitude

    # 7. Submission Generation
    # Threshold defined in the task
    THRESHOLD = 3.438959912830025

    if rmse < THRESHOLD:
        print(
            f"\nValidation metric ({rmse}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_preds = model.predict(X_test)

        # Create Submission DataFrame
        submission = pd.DataFrame({"key": test_keys, "fare_amount": test_preds})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(config.SUBMISSION_OUTPUT_PATH), exist_ok=True)

        # Save
        submission.to_csv(config.SUBMISSION_OUTPUT_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_OUTPUT_PATH}")

        # Verify file creation
        if os.path.exists(config.SUBMISSION_OUTPUT_PATH):
            print("Submission file successfully created.")
            print(submission.head())
        else:
            print("Error: Submission file was not created.")
    else:
        print(
            f"\nValidation metric ({rmse}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
