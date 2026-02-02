import os
import sys
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import from provided libraries
from library.config import Config
from library.data_pipeline import DataProcessor
from library.advanced_features import SpatiotemporalEngine
from library.model_trainer import ModelManager


def main():
    # 1. Setup and Configuration
    print("Starting Taxi Fare Prediction Pipeline...")
    np.random.seed(Config.SEED)

    # Define training sample size
    # Using None (Full Data) to maximize predictive performance (Cite solution_lesson_node_00012)
    TRAIN_SAMPLE_SIZE = None

    # 2. Data Processing (Training & Test)
    print("\n=== Phase 1: Data Processing (Training Subset) ===")
    dp = DataProcessor()

    # Load and process data.
    # We use a sample for training to meet the time constraint.
    # We load test data here as well.
    # Note: process_data returns a sampled validation set too, which we use for early stopping.
    train_df, val_sample_df, test_df = dp.process_data(
        load_cached_data=True, sample_size=TRAIN_SAMPLE_SIZE
    )

    print(f"Training Data Shape (Sampled): {train_df.shape}")
    print(f"Validation Data Shape (Sampled for Early Stopping): {val_sample_df.shape}")
    print(f"Test Data Shape: {test_df.shape}")

    # 3. Advanced Feature Engineering
    print("\n=== Phase 2: Advanced Feature Engineering ===")
    ste = SpatiotemporalEngine()

    # Fit on training sample and transform
    # We force load_cached_data=False here to ensure the TE map is built
    # specifically on our current training sample, avoiding mismatches.
    print("Fitting Spatiotemporal Features on Training Data...")
    train_df = ste.fit_transform_train(train_df, load_cached_data=False)

    # Transform validation sample and test set
    print("Transforming Validation Sample and Test Data...")
    val_sample_df = ste.transform_test(val_sample_df, load_cached_data=True)
    test_df = ste.transform_test(test_df, load_cached_data=True)

    # Define Feature Columns
    # Exclude non-feature columns (IDs, raw timestamps, target, intermediate cluster IDs)
    exclude_cols = [
        "key",
        "pickup_datetime",
        "fare_amount",
        "pickup_cluster",
        "dropoff_cluster",
        "pickup_key",
        "dropoff_key",
    ]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"Selected {len(feature_cols)} features: {feature_cols}")

    # Prepare matrices for training
    X_train = train_df[feature_cols]
    y_train = train_df["fare_amount"]
    X_val_sample = val_sample_df[feature_cols]
    y_val_sample = val_sample_df["fare_amount"]

    # 4. Model Training
    print("\n=== Phase 3: Model Training ===")
    mm = ModelManager()

    # Train XGBoost (GPU)
    print("Training XGBoost...")
    mm.train_xgboost(X_train, y_train, X_val_sample, y_val_sample)

    # Train LightGBM (CPU)
    print("Training LightGBM...")
    mm.train_lgbm(X_train, y_train, X_val_sample, y_val_sample)

    # Free up memory before full validation
    del train_df, val_sample_df, X_train, y_train, X_val_sample, y_val_sample
    gc.collect()

    # 5. Full Validation Evaluation
    print("\n=== Phase 4: Full Validation Evaluation ===")
    # Load the FULL validation dataset from metadata to meet the metric requirement
    print(f"Loading full validation set from {Config.VAL_PATH}...")
    full_val_df = dp.load_data(Config.VAL_PATH)

    # Apply pipeline to full validation set
    print("Processing full validation set...")
    full_val_df = dp.clean_data(full_val_df, mode="val")
    full_val_df = dp.add_basic_features(full_val_df)
    full_val_df = ste.transform_test(full_val_df)

    # Prepare for inference
    X_full_val = full_val_df[feature_cols]
    y_full_val = full_val_df["fare_amount"]

    # Predict
    print("Predicting on full validation set...")
    val_preds = mm.predict(X_full_val)

    # Calculate Metric
    rmse = np.sqrt(mean_squared_error(y_full_val, val_preds))
    # REQUIRED FORMAT
    print(f"Final Validation Metric: {rmse}")

    # 6. Failure Analysis
    print("\n=== Phase 5: Failure Analysis ===")
    full_val_df["predicted_fare"] = val_preds
    full_val_df["error"] = np.abs(
        full_val_df["fare_amount"] - full_val_df["predicted_fare"]
    )

    # Calculate correlations
    print("Calculating error correlations...")
    # Select numeric columns only
    numeric_df = full_val_df.select_dtypes(include=[np.number])
    correlations = numeric_df.corrwith(full_val_df["error"]).sort_values(
        ascending=False
    )

    print("Top 10 features correlated with Absolute Error:")
    print(correlations.head(10))

    # 7. Submission Generation
    print("\n=== Phase 6: Submission Generation ===")
    THRESHOLD = 3.3935366001817666

    if rmse < THRESHOLD:
        print(f"RMSE {rmse} is below threshold {THRESHOLD}. Generating submission...")

        X_test = test_df[feature_cols]
        test_preds = mm.predict(X_test)

        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_preds})

        # Save submission
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"RMSE {rmse} did not meet threshold {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
