import os
import sys
import pandas as pd
import numpy as np
import random
import logging

# Add the current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import (
    SEED,
    WORKING_DIR,
    SUBMISSION_FILE_PATH,
    COL_ID,
    COL_AFTER,
    COL_BEFORE,
    COL_SENTENCE_ID,
    COL_TOKEN_ID,
)
from library.data_loader import load_and_process_data
from library.model import HFBBModel
from library.evaluator import calculate_accuracy
from library.utils import save_submission, setup_logger


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(SEED)

    # Configure logger to be less verbose for the demo
    logger = setup_logger("Demo", level=logging.WARNING)
    print("Starting demonstration...")

    # Define a small limit for speed optimization
    DEMO_LIMIT = 50000
    VAL_LIMIT = 10000

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    print(f"\n[1/5] Loading training data (limit={DEMO_LIMIT})...")

    # Force reload from raw to demonstrate processing logic (load_cached_data=False)
    # In a real run, we would likely set this to True.
    df_train = load_and_process_data(
        split="train", load_cached_data=False, limit=DEMO_LIMIT
    )

    # Validation: Check structure
    required_cols = ["prev", "curr", "next", "after"]
    for col in required_cols:
        assert (
            col in df_train.columns
        ), f"Missing column {col} in processed training data"

    assert len(df_train) > 0, "Training dataframe is empty"
    print(f"Training data loaded. Shape: {df_train.shape}")

    # ==========================================
    # 3. Model Training Demonstration
    # ==========================================
    print("\n[2/5] Initializing and fitting HFBBModel...")
    model = HFBBModel()

    # Fit the model
    model.fit(df_train)

    # Validation: Check if model learned anything
    assert model.trigram_map is not None, "Trigram map is None after fitting"
    assert model.unigram_map is not None, "Unigram map is None after fitting"
    assert not model.trigram_map.empty, "Trigram map is empty"
    assert not model.unigram_map.empty, "Unigram map is empty"

    print(
        f"Model fitted. Trigram rules: {len(model.trigram_map)}, Unigram rules: {len(model.unigram_map)}"
    )

    # Save and Load demonstration (optional but good for verification)
    print("Testing model save/load functionality...")
    model.save()

    # Create a new instance to verify loading
    model_loaded = HFBBModel()
    loaded_successfully = model_loaded.load()
    assert loaded_successfully, "Failed to load the saved model"
    assert len(model_loaded.trigram_map) == len(
        model.trigram_map
    ), "Loaded model mismatch"

    # Use the loaded model for subsequent steps
    model = model_loaded

    # ==========================================
    # 4. Evaluation Demonstration
    # ==========================================
    print(f"\n[3/5] Evaluating on validation set (limit={VAL_LIMIT})...")

    df_val = load_and_process_data(split="val", load_cached_data=False, limit=VAL_LIMIT)

    # Predict and calculate accuracy
    accuracy = calculate_accuracy(model, df_val)

    # Validation: Accuracy bounds
    assert isinstance(accuracy, float), "Accuracy should be a float"
    assert 0.0 <= accuracy <= 1.0, f"Accuracy {accuracy} is out of bounds [0, 1]"

    print(f"Validation Accuracy on subset: {accuracy:.4f}")

    # ==========================================
    # 5. Submission Generation Demonstration
    # ==========================================
    print(f"\n[4/5] Generating submission for test set (limit={VAL_LIMIT})...")

    df_test = load_and_process_data(
        split="test", load_cached_data=False, limit=VAL_LIMIT
    )

    # Verify ID column creation in data loader
    assert COL_ID in df_test.columns, "ID column missing in test data"

    # Generate predictions
    test_preds = model.predict(df_test)

    # Validation: Prediction length
    assert len(test_preds) == len(
        df_test
    ), "Mismatch between prediction count and test data size"

    # Construct submission DataFrame
    submission_df = pd.DataFrame({COL_ID: df_test[COL_ID], COL_AFTER: test_preds})

    # Save submission
    save_submission(submission_df, filepath=SUBMISSION_FILE_PATH)
    print(f"Submission saved to {SUBMISSION_FILE_PATH}")

    # ==========================================
    # 6. Final Verification
    # ==========================================
    print("\n[5/5] Verifying submission file...")

    assert os.path.exists(SUBMISSION_FILE_PATH), "Submission file was not created"

    df_sub_check = pd.read_csv(SUBMISSION_FILE_PATH)
    assert list(df_sub_check.columns) == [
        COL_ID,
        COL_AFTER,
    ], f"Invalid columns in submission: {df_sub_check.columns}"
    assert len(df_sub_check) == len(df_test), "Submission row count mismatch"

    # Check for empty predictions (should be handled by identity backoff, but let's verify)
    # Note: 'after' can be empty string if the normalization is to remove the token,
    # but usually it's not null.
    assert (
        not df_sub_check[COL_AFTER].isnull().any()
    ), "Submission contains null values in 'after' column"

    print("Verification successful.")
    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
