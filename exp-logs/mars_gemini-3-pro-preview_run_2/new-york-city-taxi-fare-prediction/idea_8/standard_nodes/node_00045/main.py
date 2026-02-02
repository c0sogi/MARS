import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import from provided library files
from library.config import SEED, SUBMISSION_PATH
from library.data_manager import DataManager
from library.model import FarePredictor


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(SEED)
    # Suppress warnings for clean output
    warnings.filterwarnings("ignore")

    # 2. Data Loading and Preparation
    # load_cached_data=True allows using pre-computed features if available, speeding up re-runs
    # This handles the Global-Local decoupling strategy internally.
    dm = DataManager()
    X_train, y_train, X_val, y_val, X_test, test_keys = dm.load_and_prepare_data(
        load_cached_data=True
    )

    # 3. Model Training
    # The FarePredictor uses XGBoost with GPU support as configured in library/config.py
    model = FarePredictor()
    model.train(X_train, y_train, X_val, y_val)

    # 4. Validation Assessment
    # Predict on the full hold-out validation set
    val_preds = model.predict(X_val)

    # Calculate RMSE
    mse = mean_squared_error(y_val, val_preds)
    rmse = np.sqrt(mse)

    # Print the required metric string
    print(f"Final Validation Metric: {rmse}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(y_val - val_preds)

    # We calculate correlation between the error magnitude and the input features.
    # This helps identify if errors are correlated with specific features (e.g., long distances).
    # Using a copy to avoid modifying X_val
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlations
    # We drop the error_magnitude column itself from the index after correlation
    correlations = analysis_df.corrwith(analysis_df["error_magnitude"]).drop(
        "error_magnitude"
    )

    # Sort by absolute correlation to find the strongest relationships
    sorted_corrs = correlations.abs().sort_values(ascending=False)

    print("Top Feature Correlations with Error Magnitude:")
    print(sorted_corrs.head(10))

    # 6. Submission Generation
    # Threshold defined in the task
    THRESHOLD = 4.087074783740479

    if rmse < THRESHOLD:
        print(
            f"\nValidation metric ({rmse}) is strictly lower than threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions for the test set
        test_preds = model.predict(X_test)

        # Create submission DataFrame
        submission = pd.DataFrame({"key": test_keys, "fare_amount": test_preds})

        # Save to CSV
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({rmse}) is not lower than threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
