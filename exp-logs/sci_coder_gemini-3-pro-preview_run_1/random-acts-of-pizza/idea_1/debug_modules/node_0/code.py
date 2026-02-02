import os
import pandas as pd
import numpy as np
from library.config import TARGET_COL, ID_COL, SUBMISSION_PATH
from library.utils import set_seed, save_submission
from library.data_loader import load_data
from library.model import PizzaRandomForest


def run_demo():
    # 1. Set global random seed for reproducibility
    set_seed(42)

    # 2. Load Data
    # We use debug_sample_size=200 to ensure the demonstration runs quickly.
    # load_cached_data=False forces the data_loader to process the raw CSVs
    # and verify the logic in data_loader.py.
    print("Loading data (subsampled for speed)...")
    train_df, val_df, test_df = load_data(load_cached_data=False, debug_sample_size=200)

    # Verify Data Integrity
    print("Verifying data integrity...")
    assert not train_df.empty, "Training dataframe should not be empty."
    assert not val_df.empty, "Validation dataframe should not be empty."
    assert not test_df.empty, "Test dataframe should not be empty."

    # Check for essential columns
    assert (
        TARGET_COL in train_df.columns
    ), f"Target column '{TARGET_COL}' missing from training data."
    assert (
        TARGET_COL in val_df.columns
    ), f"Target column '{TARGET_COL}' missing from validation data."
    assert ID_COL in test_df.columns, f"ID column '{ID_COL}' missing from test data."
    assert (
        "combined_text" in train_df.columns
    ), "Text preprocessing in load_data failed: 'combined_text' missing."

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    # 3. Initialize Model
    # We override default parameters (n_estimators=10, max_depth=5) to optimize for speed during this demo.
    print("Initializing PizzaRandomForest model...")
    model = PizzaRandomForest(n_estimators=10, max_depth=5, random_state=42)

    # 4. Train Model
    print("Training model...")
    model.train(train_df, TARGET_COL)

    # 5. Evaluate on Validation Set
    print("Evaluating on validation set...")
    # This method prints the ROC AUC score internally
    val_preds = model.evaluate(val_df, TARGET_COL, set_name="Validation")

    # Verify Predictions
    assert isinstance(val_preds, np.ndarray), "Predictions should be a numpy array."
    assert len(val_preds) == len(
        val_df
    ), f"Prediction count ({len(val_preds)}) matches validation set size ({len(val_df)})."
    assert np.all(
        (val_preds >= 0) & (val_preds <= 1)
    ), "Probabilities must be between 0 and 1."

    # Check that predictions are not all identical (model learned something or at least varies)
    if len(np.unique(val_preds)) == 1:
        print(
            "Warning: All predictions are identical. This is expected with very small subsamples/shallow trees but checked for logic."
        )

    # 6. Generate Test Predictions
    print("Generating predictions for test set...")
    test_preds = model.predict_proba(test_df)

    assert len(test_preds) == len(test_df), "Test prediction count mismatch."

    # 7. Create Submission File
    print(f"Saving submission to {SUBMISSION_PATH}...")
    test_ids = test_df[ID_COL]
    save_submission(test_ids, test_preds, filename=SUBMISSION_PATH)

    # 8. Verify Submission File
    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created."

    submission_df = pd.read_csv(SUBMISSION_PATH)

    # Check columns
    expected_cols = [ID_COL, TARGET_COL]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}."

    # Check row count
    assert len(submission_df) == len(
        test_df
    ), f"Submission row count {len(submission_df)} does not match test set size {len(test_df)}."

    # Check value types
    assert pd.api.types.is_numeric_dtype(
        submission_df[TARGET_COL]
    ), "Target column in submission is not numeric."

    print("\nDemo completed successfully. All assertions passed.")


if __name__ == "__main__":
    run_demo()
