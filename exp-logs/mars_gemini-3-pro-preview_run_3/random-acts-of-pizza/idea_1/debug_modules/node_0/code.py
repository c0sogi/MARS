import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse

# Import from the provided library
from library.config import Config
from library.utils import set_seed, save_submission
from library.data_loader import load_dataset
from library.preprocessor import load_processed_data
from library.model import PizzaSuccessModel


def main():
    print("Starting Library Usage Demonstration...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # We modify the Config global state to ensure the demo runs quickly.
    print("\n[1] Configuring environment for rapid demonstration...")

    # Use a tiny subset of data
    DEMO_SIZE = 100
    Config.DEBUG_SAMPLE_SIZE = DEMO_SIZE

    # Reduce Random Forest complexity for speed
    Config.RF_PARAMS["n_estimators"] = 10
    Config.RF_PARAMS["max_depth"] = 5

    # Set random seed for reproducibility
    set_seed(Config.RANDOM_SEED)
    print("Configuration updated: DEBUG_SAMPLE_SIZE=100, n_estimators=10")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Loader...")

    # We explicitly set load_cached_data=False to force the loading logic to run
    train_df, val_df, test_df = load_dataset(
        load_cached_data=False, debug_size=DEMO_SIZE
    )

    # Verifications
    print(f"Loaded Train shape: {train_df.shape}")
    print(f"Loaded Val shape:   {val_df.shape}")
    print(f"Loaded Test shape:  {test_df.shape}")

    # Check if debug size was respected
    if len(train_df) != DEMO_SIZE:
        raise AssertionError(f"Expected {DEMO_SIZE} training rows, got {len(train_df)}")

    # Check if 'combined_text' feature was created
    if "combined_text" not in train_df.columns:
        raise AssertionError("'combined_text' column missing from training dataframe")

    # Check for leakage: 'number_of_upvotes_of_request_at_retrieval' should NOT be in features
    # (It is in the raw metadata but excluded by Config.NUMERICAL_COLS)
    leakage_col = "number_of_upvotes_of_request_at_retrieval"
    if leakage_col in train_df.columns:
        raise AssertionError(
            f"Leakage column '{leakage_col}' found in processed dataframe"
        )

    print("Data Loader verification passed.")

    # -------------------------------------------------------------------------
    # 3. Preprocessing Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Preprocessor...")

    # This function handles TF-IDF vectorization and stacking with numerical features.
    # We reload with load_cached_data=False to demonstrate the transformation pipeline.
    X_train, y_train, X_val, y_val, X_test, test_ids = load_processed_data(
        load_cached_data=False, debug_size=DEMO_SIZE
    )

    # Verifications
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Check dimensions
    if X_train.shape[0] != DEMO_SIZE:
        raise AssertionError(
            f"X_train has {X_train.shape[0]} rows, expected {DEMO_SIZE}"
        )

    if X_train.shape[0] != y_train.shape[0]:
        raise AssertionError("Mismatch between X_train and y_train rows")

    # Check that X is a sparse matrix
    if not scipy.sparse.issparse(X_train):
        raise AssertionError("X_train is not a sparse matrix")

    # Check feature alignment between train and test
    if X_train.shape[1] != X_test.shape[1]:
        raise AssertionError(
            f"Feature mismatch: Train has {X_train.shape[1]} cols, Test has {X_test.shape[1]} cols"
        )

    print("Preprocessor verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Training Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Model Training...")

    model = PizzaSuccessModel()

    # Train the model
    model.train(X_train, y_train, X_val, y_val)

    # Generate predictions on validation set to verify output range
    val_preds = model.predict_proba(X_val)

    # Verifications
    if len(val_preds) != len(y_val):
        raise AssertionError("Prediction length mismatch")

    if val_preds.min() < 0 or val_preds.max() > 1:
        raise AssertionError("Predictions are not valid probabilities (must be [0, 1])")

    print(
        f"Generated {len(val_preds)} predictions. Range: [{val_preds.min():.4f}, {val_preds.max():.4f}]"
    )
    print("Model training verification passed.")

    # -------------------------------------------------------------------------
    # 5. Submission Generation Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Submission Generation...")

    # Predict on test set
    test_preds = model.predict_proba(X_test)

    # Save submission
    output_path = Config.SUBMISSION_PATH
    save_submission(test_preds, test_ids, output_path)

    # Verifications
    if not os.path.exists(output_path):
        raise AssertionError(f"Submission file not created at {output_path}")

    # Read back to verify format
    sub_df = pd.read_csv(output_path)
    expected_cols = ["request_id", "requester_received_pizza"]

    if list(sub_df.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"
        )

    if len(sub_df) != DEMO_SIZE:
        raise AssertionError(
            f"Submission row count mismatch. Expected {DEMO_SIZE}, got {len(sub_df)}"
        )

    print(f"Submission file verified: {len(sub_df)} rows.")
    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
