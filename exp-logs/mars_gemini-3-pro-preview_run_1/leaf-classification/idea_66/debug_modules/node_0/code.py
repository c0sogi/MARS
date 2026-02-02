import os
import sys
import time
import numpy as np
import pandas as pd
from library import config
from library import utils
from library import feature_extraction
from library import preprocessing
from library import model


def demonstrate_utils():
    """
    Validates the utility functions for probability normalization, clipping, and scoring.
    """
    print("\n--- Demonstrating library.utils ---")

    # 1. Test normalize_probabilities
    # Create a random matrix where rows do not sum to 1
    raw_probs = np.array([[0.1, 0.2, 0.5], [2.0, 3.0, 5.0]])
    norm_probs = utils.normalize_probabilities(raw_probs)

    # Assert row sums are 1.0
    row_sums = norm_probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, err_msg="Probabilities do not sum to 1")
    print("normalize_probabilities: Verified row sums are 1.0")

    # 2. Test clip_probabilities
    # Create probabilities including 0.0 and 1.0
    extreme_probs = np.array([[0.0, 1.0], [0.5, 0.5]])
    clipped_probs = utils.clip_probabilities(extreme_probs)

    epsilon = 1e-15
    assert clipped_probs.min() >= epsilon, "Lower bound clipping failed"
    assert clipped_probs.max() <= (1.0 - epsilon), "Upper bound clipping failed"
    print("clip_probabilities: Verified clipping range [1e-15, 1-1e-15]")

    # 3. Test calculate_log_loss
    # Ground truth and good predictions
    y_true = np.array([0, 1])
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
    # Calculate loss (should be low)
    loss = utils.calculate_log_loss(y_true, y_pred, labels=[0, 1])

    # Expected loss is approx -ln(0.9) and -ln(0.8) averaged
    # -ln(0.9) ~ 0.105, -ln(0.8) ~ 0.223, avg ~ 0.164
    assert loss < 0.5, f"Log loss seems too high for good predictions: {loss}"
    print(f"calculate_log_loss: Computed loss {loss:.5f} (Expected < 0.5)")


def demonstrate_feature_extraction():
    """
    Validates the GeometricFeatureExtractor on a small subset of data.
    """
    print("\n--- Demonstrating library.feature_extraction ---")

    # Load a tiny subset of metadata to test extraction logic quickly
    if not os.path.exists(config.TRAIN_FILE):
        raise FileNotFoundError(f"Train metadata not found at {config.TRAIN_FILE}")

    train_df = pd.read_csv(config.TRAIN_FILE).head(5)

    extractor = feature_extraction.GeometricFeatureExtractor(train_df)

    print("Extracting features for 5 images (Debug Mode)...")
    start_time = time.time()
    # Use a separate cache name to avoid conflicts with the main pipeline
    features_df = extractor.extract_features(
        load_cached_data=False, cache_name="debug_geo_features.parquet", debug_limit=5
    )
    duration = time.time() - start_time
    print(f"Extraction took {duration:.4f} seconds")

    # Validation
    expected_cols = ["id"] + config.GEOMETRIC_FEATURES
    assert (
        list(features_df.columns) == expected_cols
    ), f"Columns mismatch. Expected {expected_cols}, got {list(features_df.columns)}"
    assert len(features_df) == 5, "Did not return 5 rows"

    # Check for NaNs
    assert not features_df.isnull().values.any(), "Found NaNs in extracted features"

    # Check value integrity (e.g., Solidity should be <= 1.0)
    if "Solidity" in features_df.columns:
        assert (features_df["Solidity"] <= 1.0 + 1e-9).all(), "Solidity > 1.0 detected"

    print("GeometricFeatureExtractor: Output validated successfully")


def demonstrate_preprocessing_and_model():
    """
    Validates the SanitizedGroupPreprocessor and FactorizedOASLDA model.
    Runs the full pipeline: Data Loading -> Feature Eng -> Preprocessing -> Training -> Inference.
    """
    print("\n--- Demonstrating library.preprocessing and library.model ---")

    # 1. Preprocessing
    print("Initializing SanitizedGroupPreprocessor...")
    preprocessor = preprocessing.SanitizedGroupPreprocessor()

    print(
        "Running process_and_cache() (this may take a moment for feature extraction)..."
    )
    start_time = time.time()
    data = preprocessor.process_and_cache(load_cached_data=True)
    print(f"Preprocessing completed in {time.time() - start_time:.4f} seconds")

    # Validate Data Dictionary Keys
    required_keys = [
        "X_train",
        "y_train",
        "X_val",
        "y_val",
        "X_test",
        "test_ids",
        "classes",
    ]
    for k in required_keys:
        assert k in data, f"Missing key in data dictionary: {k}"

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    classes = data["classes"]

    n_train = len(y_train)
    n_val = len(y_val)
    n_classes = len(classes)

    print(
        f"Data Loaded: {n_train} Training samples, {n_val} Validation samples, {n_classes} Classes"
    )

    # Validate Feature Groups
    expected_groups = list(config.TABULAR_FEATURE_GROUPS.keys()) + ["geometry"]
    assert set(X_train.keys()) == set(expected_groups), "Mismatch in feature groups"

    # Validate Shapes
    for group in expected_groups:
        assert (
            X_train[group].shape[0] == n_train
        ), f"Train sample count mismatch in group {group}"
        assert (
            X_val[group].shape[0] == n_val
        ), f"Val sample count mismatch in group {group}"
        assert X_train[group].shape[1] > 0, f"No features found in group {group}"
        # Check for NaNs in processed data
        assert not np.isnan(
            X_train[group]
        ).any(), f"NaNs found in processed training data for group {group}"

    # 2. Model Training
    print("Initializing FactorizedOASLDA...")
    clf = model.FactorizedOASLDA()

    print("Fitting model...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    print(f"Training completed in {time.time() - start_time:.4f} seconds")

    # 3. Validation Inference
    print("Predicting on Validation set...")
    y_pred_val = clf.predict_proba(X_val)

    # Validate Predictions
    assert y_pred_val.shape == (
        n_val,
        n_classes,
    ), f"Prediction shape mismatch. Expected {(n_val, n_classes)}, got {y_pred_val.shape}"

    # Check that probabilities sum to 1 (Softmax property)
    row_sums = y_pred_val.sum(axis=1)
    np.testing.assert_allclose(
        row_sums, 1.0, atol=1e-6, err_msg="Predictions do not sum to 1"
    )

    # 4. Scoring
    loss = utils.calculate_log_loss(y_val, y_pred_val, labels=list(range(n_classes)))
    print(f"Validation Log Loss: {loss:.5f}")

    # Baseline check: Random guessing for 99 classes is approx ln(99) = 4.6
    # The model should be significantly better than random.
    assert loss < 4.6, f"Model performance ({loss}) is worse than random guessing (4.6)"
    print("Model performance check passed.")

    # 5. Test Inference (Dry Run)
    print("Generating Test predictions...")
    X_test = data["X_test"]
    y_pred_test = clf.predict_proba(X_test)
    assert y_pred_test.shape[0] == len(
        data["test_ids"]
    ), "Test prediction count mismatch"
    print("Test inference successful.")


if __name__ == "__main__":
    # Set global seed for reproducibility
    utils.set_seed(config.SEED)

    try:
        demonstrate_utils()
        demonstrate_feature_extraction()
        demonstrate_preprocessing_and_model()
        print("\nAll demonstrations and validations passed successfully.")
    except Exception as e:
        print(f"\nFAILURE: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
