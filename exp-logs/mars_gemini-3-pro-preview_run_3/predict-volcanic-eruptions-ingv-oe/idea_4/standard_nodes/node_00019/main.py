import pandas as pd
import numpy as np
import os
import random
from sklearn.metrics import mean_absolute_error

# Import provided library modules
import library.config as config
from library.data_processor import build_dataset
from library.model_trainer import EruptionPredictor, generate_submission


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_pipeline():
    # 1. Initialization
    print("Initializing pipeline...")
    set_seed(config.SEED)

    # 2. Data Loading
    # Using load_cached_data=True to utilize pre-computed Parquet files if available
    print("Loading datasets...")
    datasets = build_dataset(load_cached_data=True)

    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["val"]
    X_test, test_ids = datasets["test"]

    print(
        f"Data loaded. Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}"
    )

    # 3. Model Training
    print("Initializing and training model...")
    # The EruptionPredictor wraps LightGBM with the config parameters
    predictor = EruptionPredictor()
    predictor.fit(X_train, y_train, X_val, y_val)

    # 4. Validation Assessment
    print("Performing validation inference...")
    val_preds = predictor.predict(X_val)

    # Calculate MAE
    mae = mean_absolute_error(y_val, val_preds)

    # Print the required metric string
    print(f"Final Validation Metric: {mae}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute errors
    abs_errors = np.abs(y_val - val_preds)

    # Create a Series for errors to compute correlation
    error_series = pd.Series(abs_errors, index=X_val.index, name="abs_error")

    # Calculate correlation between features and error magnitude
    print("Calculating feature correlations with error magnitude...")
    # corrwith returns a Series of correlations
    correlations = X_val.corrwith(error_series).abs().sort_values(ascending=False)

    print("Top 10 features most correlated with prediction error:")
    print(correlations.head(10))

    # 6. Submission Generation
    # Threshold based on previous best result
    THRESHOLD = 2739761.26

    if mae < THRESHOLD:
        print(f"\nValidation metric ({mae}) is better than threshold ({THRESHOLD}).")
        print("Generating submission file...")
        generate_submission(
            predictor, X_test, test_ids, output_path=config.SUBMISSION_PATH
        )
    else:
        print(
            f"\nValidation metric ({mae}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
