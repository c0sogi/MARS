import os
import sys
import numpy as np
import pandas as pd
from library.utils import seed_everything
from library.data_loader import LeafDataManager
from library.model_trainer import LogisticBaselineTrainer


def main():
    # 1. Setup
    print("Initializing demonstration...")
    seed_everything(42)

    metadata_dir = "./metadata"
    working_dir = "./working"
    cache_dir = os.path.join(working_dir, "demo_cache")
    submission_path = os.path.join(working_dir, "submission_demo.csv")

    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)

    # ==========================================
    # 2. Data Manager Demonstration
    # ==========================================
    print("\n--- Testing LeafDataManager ---")

    # Instantiate Data Manager
    data_manager = LeafDataManager(metadata_dir=metadata_dir, cache_dir=cache_dir)

    # Process data (loads from metadata, scales, encodes, caches)
    data_manager.process_data(load_cached_data=False)

    # Retrieve full datasets
    X_train, y_train = data_manager.get_train_data()
    X_val, y_val = data_manager.get_val_data()
    X_test, test_ids = data_manager.get_test_data()
    classes = data_manager.get_classes()

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")
    print(f"Number of Classes: {len(classes)}")

    # Verification 1: Check Feature Dimensions
    # The dataset description states there are 3 sets of 64 features = 192 features
    expected_features = 192
    assert (
        X_train.shape[1] == expected_features
    ), f"Expected {expected_features} features, got {X_train.shape[1]}"
    assert (
        X_test.shape[1] == expected_features
    ), f"Expected {expected_features} features in test set, got {X_test.shape[1]}"

    # Verification 2: Check Class Count
    # The dataset description implies 99 species
    expected_classes = 99
    assert (
        len(classes) == expected_classes
    ), f"Expected {expected_classes} classes, got {len(classes)}"

    # Verification 3: Check max_samples functionality (useful for quick debugging)
    X_subset, y_subset = data_manager.get_train_data(max_samples=10)
    assert (
        len(X_subset) == 10
    ), f"Expected 10 samples with max_samples=10, got {len(X_subset)}"

    print("Data Manager verification passed.")

    # ==========================================
    # 3. Model Trainer Demonstration
    # ==========================================
    print("\n--- Testing LogisticBaselineTrainer ---")

    # Instantiate Trainer
    trainer = LogisticBaselineTrainer(data_manager=data_manager)

    # Demonstrate Grid Search
    # We use a small list of C values to keep execution time short for this demo
    print("Running constrained grid search...")
    c_values_to_test = [0.1, 1.0, 10.0]
    best_c = trainer.grid_search_regularization(c_values=c_values_to_test)

    print(f"Best C found: {best_c}")
    assert (
        best_c in c_values_to_test
    ), "Grid search returned a value not in the search space."

    # Train Final Model
    trainer.train(c=best_c)

    # Verification 4: Check Prediction Shape and Probability Properties
    print("Verifying model predictions on validation set...")
    val_probs = trainer.predict_proba(X_val)

    assert val_probs.shape == (
        len(X_val),
        len(classes),
    ), f"Prediction shape mismatch. Expected {(len(X_val), len(classes))}, got {val_probs.shape}"

    # Check that probabilities sum to 1 (within floating point tolerance)
    row_sums = np.sum(val_probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1.0"

    print("Model Trainer verification passed.")

    # ==========================================
    # 4. Submission Generation Demonstration
    # ==========================================
    print("\n--- Testing Submission Generation ---")

    trainer.generate_submission(output_path=submission_path)

    # Verification 5: Validate the generated file
    assert os.path.exists(submission_path), "Submission file was not created."

    df_submission = pd.read_csv(submission_path)
    print(f"Loaded submission file with shape: {df_submission.shape}")

    # Check columns
    # Should have 'id' + 99 class columns = 100 columns
    assert (
        df_submission.shape[1] == 100
    ), f"Expected 100 columns in submission, got {df_submission.shape[1]}"

    assert "id" in df_submission.columns, "Column 'id' missing from submission."

    # Check row count matches test set
    assert len(df_submission) == len(
        X_test
    ), f"Expected {len(X_test)} rows in submission, got {len(df_submission)}"

    # Check value range [0, 1]
    # Drop ID column for numeric check
    numeric_data = df_submission.drop(columns=["id"]).values
    assert (
        numeric_data.min() >= 0.0 and numeric_data.max() <= 1.0
    ), "Submission contains probabilities outside [0, 1]"

    print("Submission generation verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
