import os
import numpy as np
import pandas as pd
import shutil

# Import modules from the provided library
from library.config import Config
from library.utils import log_transform, inverse_log_transform, calculate_rmsle
from library.data_manager import DataManager
from library.feature_engineer import FeaturePipeline
from library.model_trainer import RidgeRegressorWrapper


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    print("=== Starting Library Usage Demonstration ===")

    # 1. Verify Configuration and Utilities
    print("\n[1] Verifying Configuration and Utility Functions...")
    # Check directories
    assert os.path.exists(Config.INPUT_DIR), "Input directory not found."
    assert os.path.exists(Config.METADATA_DIR), "Metadata directory not found."

    # Test utility functions
    dummy_data = np.array([0.0, 1.0, 10.0])
    transformed = log_transform(dummy_data)
    inverted = inverse_log_transform(transformed)
    assert np.allclose(dummy_data, inverted), "Log transform/inverse logic failed."

    y_true = np.array([[1.0, 2.0], [1.0, 2.0]])
    y_pred = np.array([[1.1, 1.9], [1.0, 2.0]])
    score = calculate_rmsle(y_true, y_pred)
    assert score >= 0, "RMSLE calculation produced negative result."
    print("    Utils verification passed.")

    # 2. Data Loading (Training Subset)
    print("\n[2] Loading Training Data (Subset)...")
    dm = DataManager()

    # Load a small subset (50 samples) to ensure speed
    # load_cached_data=False forces the geometry loader to parse .xyz files
    train_df = dm.load_train_data(sample_size=50, load_cached_data=False)

    # Verify loaded data structure
    print(f"    Loaded {len(train_df)} training samples.")
    expected_geo_cols = Config.GEO_COLS
    for col in expected_geo_cols:
        assert (
            col in train_df.columns
        ), f"Geometry feature '{col}' missing from dataframe."

    # Check if targets exist
    for target in Config.TARGET_COLS:
        assert (
            target in train_df.columns
        ), f"Target '{target}' missing from training data."
    print("    Data loading verification passed.")

    # 3. Feature Engineering
    print("\n[3] Running Feature Engineering Pipeline...")
    pipeline = FeaturePipeline()

    # Define a temporary cache path for this demo
    train_cache_path = os.path.join(Config.WORKING_DIR, "demo_train_features.parquet")

    # Fit and transform the training data
    # is_training=True fits the encoders/scalers
    train_processed = pipeline.process_and_cache(
        train_df, cache_path=train_cache_path, is_training=True, load_cached_data=False
    )

    # Verify processed data
    # The processed dataframe should have ID, Targets, and transformed features
    assert (
        Config.ID_COL in train_processed.columns
    ), "ID column missing in processed data."
    assert not train_processed.isnull().values.any(), "Processed data contains NaNs."

    # Extract feature columns (exclude ID and Targets)
    feature_cols = [
        c
        for c in train_processed.columns
        if c not in Config.TARGET_COLS and c != Config.ID_COL
    ]

    print(f"    Generated {len(feature_cols)} features.")
    print("    Feature engineering verification passed.")

    # 4. Model Training
    print("\n[4] Training Ridge Regression Model...")
    model = RidgeRegressorWrapper()

    X_train = train_processed[feature_cols]
    y_train = train_processed[Config.TARGET_COLS]

    model.train(X_train, y_train)
    print("    Model training completed.")

    # 5. Validation
    print("\n[5] Evaluating on Validation Data (Subset)...")
    # Load a small validation subset
    val_df = dm.load_val_data(sample_size=20, load_cached_data=False)

    val_cache_path = os.path.join(Config.WORKING_DIR, "demo_val_features.parquet")

    # Transform validation data (is_training=False uses previously fitted scalers)
    val_processed = pipeline.process_and_cache(
        val_df, cache_path=val_cache_path, is_training=False, load_cached_data=False
    )

    X_val = val_processed[feature_cols]
    y_val = val_processed[Config.TARGET_COLS]

    metrics = model.evaluate(X_val, y_val)

    assert "rmsle_mean" in metrics, "Evaluation did not return RMSLE mean."
    print(f"    Validation RMSLE: {metrics['rmsle_mean']:.4f}")

    # 6. Inference on Test Data
    print("\n[6] Running Inference on Test Data (Subset)...")
    test_df = dm.load_test_data(sample_size=10, load_cached_data=False)

    test_cache_path = os.path.join(Config.WORKING_DIR, "demo_test_features.parquet")

    test_processed = pipeline.process_and_cache(
        test_df, cache_path=test_cache_path, is_training=False, load_cached_data=False
    )

    X_test = test_processed[feature_cols]

    # Predict
    predictions = model.predict(X_test)

    assert predictions.shape == (len(test_df), 2), "Prediction shape mismatch."
    assert (
        predictions >= 0
    ).all(), "Predictions contain negative values (physical impossibility)."

    print("    Sample Predictions (First 3):")
    print(predictions[:3])
    print("    Inference verification passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
