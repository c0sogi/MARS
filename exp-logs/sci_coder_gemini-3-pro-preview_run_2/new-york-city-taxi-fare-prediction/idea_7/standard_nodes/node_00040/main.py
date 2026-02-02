import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import library components
from library.data_pipeline import TaxiDataLoader
from library.model import FarePredictor
from library.config import RANDOM_STATE

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(RANDOM_STATE)

    # 2. Data Loading & Processing
    # The loader handles the Two-Stage Global-Local strategy:
    # - Global OOF Encoding on full data
    # - Subsampling and Feature Engineering
    loader = TaxiDataLoader()
    X_train, y_train, X_val, y_val, X_test, test_keys = loader.get_processed_data(
        load_cached_data=True
    )

    # 3. Model Training
    # Initialize predictor with default config (XGBoost on GPU)
    predictor = FarePredictor()

    # Train the model
    # Using default rounds from config, which includes early stopping
    predictor.train(X_train, y_train, X_val, y_val)

    # 4. Validation Assessment
    # Predict on validation set
    val_preds = predictor.predict(X_val)

    # Calculate RMSE
    mse = mean_squared_error(y_val, val_preds)
    rmse = np.sqrt(mse)

    # Print required metric
    print(f"Final Validation Metric: {rmse}")

    # 5. Failure Analysis
    print("Performing failure analysis...")
    # Calculate absolute residuals
    residuals = np.abs(y_val - val_preds)

    # Create a temporary dataframe for correlation analysis
    # We use the validation features and append the error magnitude
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = residuals

    # Compute correlation of features with error magnitude
    # Drop the error column itself from the index
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

    # Sort by absolute correlation strength
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    print("Top 5 Feature Correlations with Error Magnitude:")
    print(top_correlations)

    # 6. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 4.087074783740479

    if rmse < THRESHOLD:
        print(
            f"Validation metric {rmse} meets threshold {THRESHOLD}. Generating submission..."
        )

        # Generate predictions for test set
        test_preds = predictor.predict(X_test)

        # Save submission
        predictor.save_submission(test_keys, test_preds)
    else:
        print(
            f"Validation metric {rmse} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
