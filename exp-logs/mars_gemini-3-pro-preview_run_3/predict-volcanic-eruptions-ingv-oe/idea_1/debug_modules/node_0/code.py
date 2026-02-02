import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import (
    SEED,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    SUBMISSION_PATH,
    LGB_PARAMS,
)
from library.dataset import load_dataset
from library.model import EruptionPredictor
from library.features import generate_features


# Ensure reproducibility
def set_seed(seed=SEED):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("Initializing demonstration...")
    set_seed()

    # ==========================================
    # 1. Data Loading & Feature Extraction
    # ==========================================
    # We use a small debug_size to ensure the script runs quickly (within minutes).
    # load_cached_data=False ensures we demonstrate the raw feature extraction logic.
    DEBUG_SIZE = 50

    print(f"Loading training data (subset of {DEBUG_SIZE} samples)...")
    X_train, y_train = load_dataset(
        metadata_path=TRAIN_META_PATH,
        is_train=True,
        load_cached_data=False,
        debug_size=DEBUG_SIZE,
    )

    print(f"Loading validation data (subset of {DEBUG_SIZE} samples)...")
    X_val, y_val = load_dataset(
        metadata_path=VAL_META_PATH,
        is_train=True,
        load_cached_data=False,
        debug_size=DEBUG_SIZE,
    )

    print(f"Loading test data (subset of {DEBUG_SIZE} samples)...")
    X_test, _ = load_dataset(
        metadata_path=TEST_META_PATH,
        is_train=False,
        load_cached_data=False,
        debug_size=DEBUG_SIZE,
    )

    # ==========================================
    # 2. Data Verification
    # ==========================================
    print("Verifying data integrity...")

    # Check shapes
    assert (
        len(X_train) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} training samples, got {len(X_train)}"
    assert (
        len(X_val) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} validation samples, got {len(X_val)}"
    assert (
        len(X_test) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} test samples, got {len(X_test)}"

    # Check targets
    assert y_train is not None, "Training target (y_train) should not be None"
    assert len(y_train) == len(X_train), "Mismatch between X_train and y_train length"
    assert not y_train.isnull().any(), "y_train contains null values"

    # Check required columns
    assert "segment_id" in X_train.columns, "segment_id missing from X_train"
    assert "segment_id" in X_test.columns, "segment_id missing from X_test"

    # Check feature generation (approximate check based on number of sensors and stats)
    # 10 sensors * ~13 stats each = ~130 columns + segment_id
    assert (
        X_train.shape[1] > 100
    ), f"Feature extraction seems incomplete. Only {X_train.shape[1]} columns found."

    print("Data verification passed.")

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("Initializing model...")
    predictor = EruptionPredictor()

    # OPTIMIZATION FOR SPEED:
    # Override the default configuration to run a very fast training cycle.
    # Default is 2000 estimators; we reduce to 20 for this demo.
    predictor.model.set_params(n_estimators=20, verbose=-1)

    print("Training model...")
    predictor.fit(X_train, y_train, X_val, y_val)

    # Verify model state
    assert (
        predictor.feature_names is not None
    ), "Feature names were not captured during fitting."
    assert len(predictor.feature_names) == (
        X_train.shape[1] - 1
    ), "Feature count mismatch (excluding segment_id)."

    # ==========================================
    # 4. Prediction & Submission
    # ==========================================
    print("Generating predictions on test set...")
    predictions = predictor.predict(X_test)

    # Verify predictions
    assert len(predictions) == len(X_test), "Prediction length mismatch."
    assert np.issubdtype(predictions.dtype, np.number), "Predictions are not numeric."
    assert not np.isnan(predictions).any(), "Predictions contain NaNs."

    print("Creating submission file...")
    predictor.create_submission(X_test, predictions)

    # Verify submission file
    assert os.path.exists(
        SUBMISSION_PATH
    ), f"Submission file not found at {SUBMISSION_PATH}"

    submission_df = pd.read_csv(SUBMISSION_PATH)
    assert list(submission_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Submission columns are incorrect."
    assert (
        len(submission_df) == DEBUG_SIZE
    ), f"Submission row count mismatch. Expected {DEBUG_SIZE}, got {len(submission_df)}"

    print("\nDemonstration completed successfully.")
    print(f"Output saved to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
