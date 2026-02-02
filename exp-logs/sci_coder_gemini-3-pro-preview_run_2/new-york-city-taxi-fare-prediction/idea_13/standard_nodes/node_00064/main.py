import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Import provided libraries
from library.config import RANDOM_SEED, SUBMISSION_PATH
from library.data_manager import DataManager
from library.model_trainer import ModelTrainer


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_pipeline():
    # 1. Initialization
    print("Starting Pipeline Execution...")
    set_seed(RANDOM_SEED)

    # Initialize Managers
    data_manager = DataManager()
    model_trainer = ModelTrainer()

    # 2. Data Preparation
    # Load and process training data (includes K-Fold Spatial Priors and Subsampling)
    print("\n--- Step 1: Preparing Training Data ---")
    train_df = data_manager.prepare_training_data(load_cached_data=True)

    # Load and process validation data (applies Global Priors)
    print("\n--- Step 2: Preparing Validation Data ---")
    # Note: If cached data exists, full_train_df is not needed.
    # If not, DataManager handles loading it internally.
    val_df = data_manager.prepare_validation_data(load_cached_data=True)

    # 3. Model Training
    print("\n--- Step 3: Training Model ---")
    model = model_trainer.train_xgboost(train_df, val_df)

    # Clean up training data to free memory
    del train_df
    import gc

    gc.collect()

    # 4. Evaluation
    print("\n--- Step 4: Evaluation ---")
    # Prepare validation features
    target_col = "fare_amount"
    exclude_cols = [target_col, "key"]
    X_val = val_df.drop(columns=[c for c in exclude_cols if c in val_df.columns])
    y_val = val_df[target_col]

    # Inference on Validation Set
    val_preds = model.predict(X_val)

    # Apply Post-Processing (Floor at $2.50)
    val_preds = np.maximum(val_preds, 2.50)

    # Calculate RMSE
    final_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    # Print Required Metric
    print(f"Final Validation Metric: {final_rmse}")

    # 5. Failure Analysis
    print("\n--- Step 5: Failure Analysis ---")
    # Calculate residuals (Absolute Error)
    residuals = np.abs(y_val - val_preds)

    # Calculate correlation between features and residuals
    # Select numerical columns only
    numerical_features = X_val.select_dtypes(include=[np.number])
    correlations = numerical_features.corrwith(residuals).sort_values(ascending=False)

    print("Top Feature Correlations with Error (Residuals):")
    print(correlations.head(10))

    # 6. Submission Generation
    THRESHOLD = 3.5069767944123895

    if final_rmse < THRESHOLD:
        print(
            f"\nMetric {final_rmse} is better than threshold {THRESHOLD}. Generating submission..."
        )

        # Prepare Test Data
        print("\n--- Step 6: Preparing Test Data ---")
        test_df = data_manager.prepare_test_data(load_cached_data=True)

        # Generate Predictions
        test_preds = model_trainer.predict_and_postprocess(model, test_df)

        # Save Submission
        model_trainer.generate_submission(test_df, test_preds)

    else:
        print(
            f"\nMetric {final_rmse} did not meet threshold {THRESHOLD}. Skipping submission."
        )

    print("\nPipeline Execution Complete.")


if __name__ == "__main__":
    run_pipeline()
