import os
import sys
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library import config
from library import utils
from library.data_processing import DataHandler
from library import train_eval
from library import model as lib_model


def main():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Set Random Seed for Reproducibility
    # Using the utility function to ensure consistent behavior
    utils.set_seed(42)
    print("Random seed set.")

    # 2. Data Processing Demonstration
    print("\n--- Data Processing ---")

    # Instantiate the DataHandler class
    data_handler = DataHandler()

    # Load and process data
    # This will handle loading CSVs, feature engineering (splitting f_27),
    # preprocessing (scaling/encoding), and caching.
    # We enable caching to speed up subsequent runs if this script were run multiple times.
    print("Retrieving processed data...")
    X_train, y_train, X_val, y_val, X_test, ids_test = data_handler.get_processed_data(
        load_cached_data=True
    )

    # Verify Data Shapes
    # Based on metadata: Train=640k, Val=160k, Test=100k
    print("Verifying data shapes...")
    assert (
        X_train.shape[0] == 640000
    ), f"Expected 640,000 training samples, got {X_train.shape[0]}"
    assert (
        y_train.shape[0] == 640000
    ), f"Expected 640,000 training targets, got {y_train.shape[0]}"
    assert (
        X_val.shape[0] == 160000
    ), f"Expected 160,000 validation samples, got {X_val.shape[0]}"
    assert (
        y_val.shape[0] == 160000
    ), f"Expected 160,000 validation targets, got {y_val.shape[0]}"
    assert (
        X_test.shape[0] == 100000
    ), f"Expected 100,000 test samples, got {X_test.shape[0]}"
    assert (
        ids_test.shape[0] == 100000
    ), f"Expected 100,000 test IDs, got {ids_test.shape[0]}"

    # Verify Feature Expansion
    # Original numerical cols (30) + OneHotEncoded f_27 (10 positions * ~26 chars)
    n_features = X_train.shape[1]
    print(f"Total features after processing: {n_features}")
    assert (
        n_features > 30
    ), "Feature count suggests f_27 was not expanded/encoded correctly."

    print("Data processing verification passed.")

    # 3. Model Training Demonstration
    print("\n--- Model Training ---")

    # We train on a small subset to demonstrate functionality quickly.
    # train_eval.train_model handles model instantiation and fitting.
    subset_size = 5000
    print(f"Training Logistic Regression on a subset of {subset_size} samples...")

    model = train_eval.train_model(
        X_train,
        y_train,
        max_iter=50,  # Reduced iterations for speed
        max_samples=subset_size,  # Subset for speed
    )

    # Verify Model Artifact
    # The train_model function should save the model to config.MODEL_PATH
    assert os.path.exists(
        config.MODEL_PATH
    ), f"Model file was not created at {config.MODEL_PATH}"
    print("Model training complete and artifact verified.")

    # 4. Evaluation Demonstration
    print("\n--- Evaluation ---")

    # Evaluate on a subset of the validation set
    val_subset_size = 5000
    print(f"Evaluating on validation subset ({val_subset_size} samples)...")

    auc_score = train_eval.evaluate_model(
        model, X_val[:val_subset_size], y_val[:val_subset_size]
    )

    print(f"Validation AUC: {auc_score:.4f}")

    # Verify Metric Range
    assert (
        0.0 <= auc_score <= 1.0
    ), f"AUC score {auc_score} is out of valid range [0, 1]"
    print("Evaluation verification passed.")

    # 5. Prediction / Submission Demonstration
    print("\n--- Submission Generation ---")

    # Generate predictions for the full test set
    # train_eval.predict_test calls library.model.generate_submission internally
    print("Generating predictions for test set...")
    train_eval.predict_test(model, X_test, ids_test)

    # Verify Submission File
    assert os.path.exists(
        config.SUBMISSION_PATH
    ), f"Submission file not found at {config.SUBMISSION_PATH}"

    # Check file content format
    df_submission = pd.read_csv(config.SUBMISSION_PATH)

    # Check dimensions
    assert df_submission.shape == (
        100000,
        2,
    ), f"Submission shape mismatch: {df_submission.shape}"

    # Check columns
    expected_cols = ["id", "target"]
    assert (
        list(df_submission.columns) == expected_cols
    ), f"Column mismatch. Expected {expected_cols}, got {list(df_submission.columns)}"

    # Check ID consistency
    assert (
        df_submission["id"].iloc[0] == ids_test[0]
    ), "First ID in submission does not match test data."

    # Check probability range
    preds = df_submission["target"]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions contain values outside [0, 1]"

    print(f"Submission file verified at: {config.SUBMISSION_PATH}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
