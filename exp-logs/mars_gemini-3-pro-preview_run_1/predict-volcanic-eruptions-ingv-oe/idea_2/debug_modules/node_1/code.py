import os
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import (
    TRAIN_METADATA_PATH,
    TEST_METADATA_PATH,
    MODEL_CONFIG,
    SUBMISSION_PATH,
    WORKING_DIR,
)
from library.features import FeatureManager
from library.model import CrossValidator, InferenceModel
from library.utils import seed_everything


def main():
    # 1. Setup
    print("Initializing demonstration...")
    seed_everything(42)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # 2. Demonstrate Feature Extraction on a Small Subset
    print("\n--- Step 1: Feature Extraction (Subset) ---")

    # Load raw metadata
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {TRAIN_METADATA_PATH}")

    full_train_meta = pd.read_csv(TRAIN_METADATA_PATH)

    # Select a small subset for speed (20 samples)
    subset_size = 20
    small_train_meta = full_train_meta.head(subset_size).copy()

    print(f"Processing {subset_size} training samples...")

    # Initialize FeatureManager
    fm = FeatureManager()

    # Extract features
    # We set load_cached_data=False to force computation and demonstrate the extraction logic.
    # Note: This will overwrite the cache file in WORKING_DIR with this subset.
    X_train, y_train = fm.get_train_data(small_train_meta, load_cached_data=False)

    # Validation
    print(f"Extracted features shape: {X_train.shape}")
    assert (
        len(X_train) == subset_size
    ), f"Expected {subset_size} rows, got {len(X_train)}"
    assert (
        len(y_train) == subset_size
    ), f"Expected {subset_size} targets, got {len(y_train)}"
    assert not X_train.isnull().values.any(), "Features contain NaNs"

    # Check if specific features exist (based on default config)
    # Example: sensor_1 -> mean (window) -> mean (agg)
    expected_col_part = "sensor_1_mean_mean"
    assert (
        expected_col_part in X_train.columns
    ), f"Missing expected feature {expected_col_part}"

    print("Feature extraction verified.")

    # 3. Demonstrate Model Training (Cross-Validation)
    print("\n--- Step 2: Model Training (Fast Mode) ---")

    # Modify config for speed and small data compatibility
    fast_model_config = MODEL_CONFIG.copy()
    fast_model_config.update(
        {
            "n_estimators": 10,  # Very few trees for speed
            "early_stopping_rounds": 5,
            "verbose": -1,
            "verbosity": -1,
            # With 20 samples and 5 folds, train set is 16.
            # min_data_in_leaf must be <= 16.
            "min_data_in_leaf": 2,
            "bagging_freq": 1,  # Ensure bagging happens
        }
    )

    # Initialize CrossValidator with the fast config
    cv = CrossValidator(config=fast_model_config)

    # Train
    scores, overall_mae = cv.train(X_train, y_train)

    # Validation
    print(f"Fold Scores (MAE): {scores}")
    print(f"Overall CV MAE: {overall_mae}")

    assert len(scores) == 5, "Did not receive scores for 5 folds"
    assert overall_mae >= 0, "MAE cannot be negative"

    # Check if models were saved
    expected_model_path = os.path.join(WORKING_DIR, "lgb_model_fold_0.txt")
    assert os.path.exists(
        expected_model_path
    ), f"Model file not found at {expected_model_path}"

    print("Model training verified.")

    # 4. Demonstrate Inference
    print("\n--- Step 3: Inference and Submission ---")

    # Load test metadata subset
    full_test_meta = pd.read_csv(TEST_METADATA_PATH)
    small_test_meta = full_test_meta.head(5).copy()

    # Extract test features
    print(f"Processing {len(small_test_meta)} test samples...")
    X_test, _ = fm.get_test_data(small_test_meta, load_cached_data=False)

    # Initialize InferenceModel (loads models from WORKING_DIR)
    inference_model = InferenceModel(model_dir=WORKING_DIR)

    # Predict
    predictions = inference_model.predict(X_test)

    # Validation
    assert len(predictions) == len(small_test_meta), "Prediction count mismatch"
    assert np.all(np.isfinite(predictions)), "Predictions contain non-finite values"

    print(f"Sample predictions: {predictions}")

    # Generate Submission
    segment_ids = small_test_meta["segment_id"]
    inference_model.generate_submission(X_test, segment_ids)

    # Verify submission file
    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(SUBMISSION_PATH)
    assert len(df_sub) == len(
        small_test_meta
    ), "Submission file has incorrect number of rows"
    assert list(df_sub.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Incorrect columns in submission"

    print(f"Submission saved to {SUBMISSION_PATH}")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
