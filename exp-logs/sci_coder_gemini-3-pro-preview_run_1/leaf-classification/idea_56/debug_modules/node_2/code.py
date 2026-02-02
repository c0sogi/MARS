import os
import numpy as np
import pandas as pd
from library.utils import set_seed, validate_paths
from library.feature_extraction import load_data_with_features
from library.pipeline import get_preprocessed_data, SanitizedPreprocessor
from library.model import OASLinearDiscriminant, train_and_evaluate, generate_submission
from library.config import SUBMISSION_DIR, SEED

# 1. Setup and Configuration
print(">>> Step 1: Initialization")
set_seed(SEED)
validate_paths()
print("Initialization complete.\n")

if __name__ == "__main__":
    # Define a small limit for demonstration speed
    DEMO_LIMIT = 50

    # 2. Feature Extraction Demonstration
    print(f">>> Step 2: Feature Extraction (Limit: {DEMO_LIMIT} samples)")
    # We load a small subset of the training data to demonstrate the Hybrid Geometric Fusion
    X_raw, y_raw, ids_raw = load_data_with_features(
        dataset_type="train",
        load_cached_data=False,  # Force computation to demonstrate the extractor
        limit=DEMO_LIMIT,
    )

    # Validation: Check shapes and content
    assert len(X_raw) == DEMO_LIMIT, f"Expected {DEMO_LIMIT} rows, got {len(X_raw)}"
    assert len(y_raw) == DEMO_LIMIT, "Target vector length mismatch"

    # Check if geometric features were added
    expected_geo_col = "geo_area"
    assert (
        expected_geo_col in X_raw.columns
    ), f"Missing geometric feature: {expected_geo_col}"

    # Check if tabular features exist (e.g., margin1)
    assert "margin1" in X_raw.columns, "Missing tabular feature: margin1"

    print(f"Successfully extracted features. Matrix shape: {X_raw.shape}")
    print("Feature extraction verification passed.\n")

    # 3. Pipeline Processing Demonstration
    print(f">>> Step 3: Pipeline Processing (Sanitized Preprocessor)")
    # Load preprocessed data (this handles splitting, sanitization, power transform, scaling)
    # We use a fresh limit to ensure the pipeline runs fully
    X_train, y_train, X_val, y_val, X_test, test_ids = get_preprocessed_data(
        load_cached_data=False,
        limit=DEMO_LIMIT * 2,  # Slightly larger to ensure train/val split has data
    )

    # Validation: Check shapes
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    assert (
        X_train.shape[1] == X_val.shape[1]
    ), "Feature count mismatch between train and val"
    assert not np.isnan(X_train).any(), "NaN values found in preprocessed training data"

    # Validation: Check Preprocessor Logic directly
    print("Verifying SanitizedPreprocessor logic...")
    preprocessor = SanitizedPreprocessor()
    # Create dummy data with a constant column to test VarianceThreshold
    dummy_X = np.random.rand(20, 5)
    dummy_X[:, 0] = 1.0  # Constant column

    preprocessor.fit(dummy_X)
    dummy_trans = preprocessor.transform(dummy_X)

    # The constant column (index 0) should be removed, so output width should be 4
    assert (
        dummy_trans.shape[1] == 4
    ), f"VarianceThreshold failed. Expected 4 features, got {dummy_trans.shape[1]}"

    print("Pipeline verification passed.\n")

    # 4. Model Training and Evaluation
    print(">>> Step 4: Model Training (OAS Linear Discriminant)")

    # Train the model
    model = train_and_evaluate(X_train, y_train, X_val, y_val)

    # Validation: Check Model Attributes
    assert isinstance(model, OASLinearDiscriminant), "Model is not of expected type"
    assert hasattr(model, "coef_"), "Model missing coef_ attribute"
    assert hasattr(model, "intercept_"), "Model missing intercept_ attribute"

    # Validation: Check Predictions
    # Predict on a few validation samples
    sample_probs = model.predict_proba(X_val[:5])

    # Check probability range [0, 1]
    assert (sample_probs >= 0).all() and (
        sample_probs <= 1
    ).all(), "Probabilities out of range"

    # Check summation to 1 (allow small float error)
    row_sums = np.sum(sample_probs, axis=1)
    assert np.allclose(row_sums, 1.0), f"Probabilities do not sum to 1: {row_sums}"

    print("Model training and verification passed.\n")

    # 5. Submission Generation
    print(">>> Step 5: Submission Generation")

    # Generate submission file
    generate_submission(model, X_test, test_ids)

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Validation: Check file existence and format
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Check columns
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    # Check that we have columns for classes
    assert len(df_sub.columns) > 2, "Submission seems to lack class columns"
    # Check row count matches test set
    assert len(df_sub) == len(
        test_ids
    ), f"Submission row count mismatch. Expected {len(test_ids)}, got {len(df_sub)}"

    print("Submission generation verification passed.")
    print("\n>>> All demonstration steps completed successfully.")
