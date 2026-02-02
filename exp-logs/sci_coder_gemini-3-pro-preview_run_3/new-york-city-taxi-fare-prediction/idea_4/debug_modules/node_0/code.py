import os
import sys
import numpy as np
import pandas as pd
import warnings
import joblib

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.data_pipeline import DataProcessor
from library.advanced_features import SpatiotemporalEngine
from library.model_trainer import ModelManager


def main():
    # 1. Setup and Configuration Overrides
    print("=== Setting up demonstration environment ===")

    # Set seeds for reproducibility
    np.random.seed(42)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Override Config parameters for speed
    print("Overriding configuration for fast demonstration...")
    Config.XGB_PARAMS["n_estimators"] = 20
    Config.XGB_PARAMS["early_stopping_rounds"] = 5

    Config.LGBM_PARAMS["n_estimators"] = 20
    Config.LGBM_PARAMS["early_stopping_rounds"] = 5

    Config.KMEANS_CLUSTERS = 20  # Reduced from 500
    Config.TE_FOLDS = 3  # Reduced from 5

    # Define a sample size for this demo
    DEMO_SAMPLE_SIZE = 5000

    # 2. Data Pipeline Demonstration
    print("\n=== Running Data Pipeline (DataProcessor) ===")
    processor = DataProcessor()

    # Process data with sampling. We set load_cached_data=False to force execution of the logic.
    train_df, val_df, test_df = processor.process_data(
        load_cached_data=False, sample_size=DEMO_SAMPLE_SIZE
    )

    # Validation: Check basic feature generation
    print("Validating DataProcessor outputs...")
    expected_cols = ["haversine_dist", "pickup_lon_rot", "hour", "year"]
    for col in expected_cols:
        if col not in train_df.columns:
            raise AssertionError(f"Missing expected feature '{col}' in training data.")

    # Check that rows were filtered/sampled correctly
    if len(train_df) > DEMO_SAMPLE_SIZE:
        raise AssertionError(
            f"Training set size {len(train_df)} exceeds sample limit {DEMO_SAMPLE_SIZE}"
        )

    # Check cleaning (fare amount range)
    if (
        train_df["fare_amount"].min() < Config.FARE_MIN
        or train_df["fare_amount"].max() > Config.FARE_MAX
    ):
        raise AssertionError("Fare amount outliers were not filtered correctly.")

    print("DataProcessor validation successful.")

    # 3. Advanced Features Demonstration
    print("\n=== Running Spatiotemporal Engine ===")
    engine = SpatiotemporalEngine()

    # Fit and transform training data
    # We force re-computation to demonstrate the logic
    train_df = engine.fit_transform_train(train_df, load_cached_data=False)

    # Transform validation and test data
    val_df = engine.transform_test(val_df, load_cached_data=False)
    test_df = engine.transform_test(test_df, load_cached_data=False)

    # Validation: Check advanced feature generation
    print("Validating SpatiotemporalEngine outputs...")
    te_cols = ["te_pickup", "te_dropoff", "pickup_cluster", "dropoff_cluster"]
    for col in te_cols:
        if col not in train_df.columns:
            raise AssertionError(
                f"Missing Spatiotemporal feature '{col}' in training data."
            )
        if col not in test_df.columns:
            raise AssertionError(
                f"Missing Spatiotemporal feature '{col}' in test data."
            )

    # Check for NaNs in Target Encoded columns (should be filled by global mean)
    if train_df["te_pickup"].isnull().any():
        raise AssertionError(
            "NaN values found in 'te_pickup' column after transformation."
        )

    print("SpatiotemporalEngine validation successful.")

    # 4. Model Training Demonstration
    print("\n=== Running Model Manager ===")
    manager = ModelManager()

    # Prepare Feature Matrix
    # Exclude non-feature columns
    exclude_cols = ["key", "fare_amount", "pickup_datetime"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Training with {len(feature_cols)} features: {feature_cols}")

    X_train = train_df[feature_cols]
    y_train = train_df["fare_amount"]

    X_val = val_df[feature_cols]
    y_val = val_df["fare_amount"]

    X_test = test_df[feature_cols]

    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = manager.train_xgboost(X_train, y_train, X_val, y_val)

    # Train LightGBM
    print("Training LightGBM...")
    lgbm_model = manager.train_lgbm(X_train, y_train, X_val, y_val)

    # Validation: Check model persistence
    if not os.path.exists(
        Config.XGB_PARAMS["device"] == "cuda" and manager.xgb_path or manager.xgb_path
    ):
        # Note: path check is sufficient
        pass

    if not os.path.exists(manager.xgb_path):
        raise AssertionError(f"XGBoost model file not found at {manager.xgb_path}")
    if not os.path.exists(manager.lgbm_path):
        raise AssertionError(f"LightGBM model file not found at {manager.lgbm_path}")

    # 5. Prediction and Submission
    print("\n=== Generating Predictions ===")
    predictions = manager.predict(X_test)

    # Validation: Check predictions
    if len(predictions) != len(test_df):
        raise AssertionError("Prediction length mismatch.")

    if np.isnan(predictions).any():
        raise AssertionError("NaN values found in predictions.")

    print(f"Predictions generated. Mean Fare: ${predictions.mean():.2f}")

    # Create Submission File
    submission = pd.DataFrame({"key": test_df["key"], "fare_amount": predictions})

    # Ensure submission directory exists (handled by Config, but good to double check logic)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Final check of the file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not created.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
