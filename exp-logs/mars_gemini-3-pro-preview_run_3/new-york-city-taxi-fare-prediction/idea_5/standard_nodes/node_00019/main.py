import sys
import os
import numpy as np
import pandas as pd
import random
import warnings
from sklearn.metrics import mean_squared_error

# Import from provided library files
from library.config import RANDOM_SEED
from library.data_loader import TaxiDataLoader
from library.model_trainer import ModelTrainer


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)
    warnings.filterwarnings("ignore")
    print("Starting pipeline execution...")

    # 2. Data Loading
    # We use debug_mode=False to ensure we access the real data structure,
    # but we will manually sample for the 'fast baseline' requirement.
    loader = TaxiDataLoader(debug_mode=False, load_cached_data=True)

    print("Loading training data...")
    X_train, y_train = loader.get_train_data()

    # Fast Baseline Sampling
    # Limit training data to 1,000,000 rows to ensure completion within 2 hours
    MAX_TRAIN_SAMPLES = 1_000_000
    if len(X_train) > MAX_TRAIN_SAMPLES:
        print(
            f"Sampling training data from {len(X_train)} to {MAX_TRAIN_SAMPLES} rows for fast baseline..."
        )
        indices = np.random.choice(len(X_train), MAX_TRAIN_SAMPLES, replace=False)
        X_train = X_train.iloc[indices].reset_index(drop=True)
        y_train = y_train.iloc[indices].reset_index(drop=True)

    print("Loading validation data...")
    # We use the full validation set for accurate metric calculation
    X_val, y_val = loader.get_val_data()

    # 3. Model Training
    trainer = ModelTrainer()

    # Train XGBoost (GPU)
    # forcing load_cached_model=False to ensure we train on the current sampled data
    trainer.train_xgboost(X_train, y_train, X_val, y_val, load_cached_model=False)

    # Train LightGBM (CPU)
    trainer.train_lgbm(X_train, y_train, X_val, y_val, load_cached_model=False)

    # 4. Validation Assessment
    print("Evaluating ensemble on validation set...")
    val_preds = trainer.predict_ensemble(X_val)

    # Calculate RMSE
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Final Validation Metric: {val_rmse}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(y_val - val_preds)

    # Create analysis dataframe
    # We copy X_val to avoid modifying the original
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = errors

    # Calculate correlations
    print("Calculating error correlations...")
    correlations = (
        analysis_df.corrwith(analysis_df["error_magnitude"])
        .abs()
        .sort_values(ascending=False)
    )

    print("Top 5 features correlated with prediction error:")
    print(correlations.drop("error_magnitude", errors="ignore").head(5))

    # 6. Submission Generation
    THRESHOLD = 3.3935366001817666

    if val_rmse < THRESHOLD:
        print(
            f"\nValidation metric ({val_rmse}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        X_test, keys = loader.get_test_data()

        # Predict
        test_preds = trainer.predict_ensemble(X_test)

        # Save
        trainer.save_submission(keys, test_preds)
    else:
        print(
            f"\nValidation metric ({val_rmse}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
