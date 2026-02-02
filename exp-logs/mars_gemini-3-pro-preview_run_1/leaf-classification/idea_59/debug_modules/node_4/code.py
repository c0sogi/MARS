import os
import sys
import numpy as np
import pandas as pd
from library import config, utils, data_handler, preprocessor, model


def main():
    print("=== Leaf Classification Pipeline Demo ===")

    # 1. Setup and Reproducibility
    print("\n[Step 1] Setting up environment...")
    utils.set_seed(config.SEED)

    # We use a small debug size to ensure the demo runs quickly (< 1 min)
    DEBUG_SIZE = 20
    print(f"Debug sample size set to: {DEBUG_SIZE}")

    # 2. Data Loading Demonstration
    print("\n[Step 2] Testing Data Handler...")

    # Load Train
    X_train, y_train, ids_train = data_handler.load_dataset(
        "train", debug_size=DEBUG_SIZE, load_cached_data=False
    )

    # Load Val
    X_val, y_val, ids_val = data_handler.load_dataset(
        "val", debug_size=DEBUG_SIZE, load_cached_data=False
    )

    # Load Test
    X_test, y_test, ids_test = data_handler.load_dataset(
        "test", debug_size=DEBUG_SIZE, load_cached_data=False
    )

    # Validation Checks for Data Loading
    print("Validating loaded data shapes and types...")

    # Check sample counts
    # Cite debug_lesson_6: Dynamic Subsampling Requires Dynamic Assertions
    assert (
        len(X_train) <= DEBUG_SIZE and len(X_train) > 0
    ), f"Expected <= {DEBUG_SIZE} training samples, got {len(X_train)}"
    assert (
        len(X_val) <= DEBUG_SIZE and len(X_val) > 0
    ), f"Expected <= {DEBUG_SIZE} validation samples, got {len(X_val)}"
    assert (
        len(X_test) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} test samples, got {len(X_test)}"

    # Check feature count: 192 tabular (3 sets * 64) + 9 geometric
    expected_features = 192 + 9
    assert (
        X_train.shape[1] == expected_features
    ), f"Expected {expected_features} features, got {X_train.shape[1]}"

    # Check IDs
    assert len(ids_train) == DEBUG_SIZE
    assert ids_train.dtype == np.int64 or ids_train.dtype == int

    # Check Targets
    assert len(y_train) == DEBUG_SIZE
    assert y_test is None, "Test set should not have targets"

    print("Data loading verification passed.")

    # 3. Preprocessing Demonstration
    print("\n[Step 3] Testing Preprocessor...")

    # We force re-computation to demonstrate the logic, though caching is supported
    X_train_trans, X_val_trans, X_test_trans = preprocessor.get_transformed_data(
        X_train,
        X_val,
        X_test,
        debug_suffix=f"_debug_{DEBUG_SIZE}",
        load_cached_data=False,
    )

    # Validation Checks for Preprocessing
    print("Validating transformed data...")

    # Check types (Must be float64 as per config)
    assert (
        X_train_trans.dtype == config.FLOAT_PRECISION
    ), "Transformed data must be float64"

    # Check for NaNs
    assert not np.isnan(X_train_trans).any(), "Transformed training data contains NaNs"
    assert not np.isnan(X_val_trans).any(), "Transformed validation data contains NaNs"

    # Check that feature count might have changed due to VarianceThreshold (though unlikely with 0 threshold on continuous data)
    # But rows should remain same
    assert X_train_trans.shape[0] == len(X_train)

    print("Preprocessing verification passed.")

    # 4. Model Demonstration
    print("\n[Step 4] Testing OASLinearModel...")

    clf = model.OASLinearModel()

    # Fit
    clf.fit(X_train_trans, y_train)

    # Validate Model Internals
    n_classes = len(np.unique(y_train))
    n_features = X_train_trans.shape[1]

    assert clf.W.shape == (n_classes, n_features), "Weight matrix shape mismatch"
    assert clf.b.shape == (n_classes,), "Bias vector shape mismatch"
    assert clf.shrinkage_ is not None, "OAS shrinkage not computed"
    print(f"Model fitted. OAS Shrinkage: {clf.shrinkage_:.4f}")

    # Predict Proba
    val_probs = clf.predict_proba(X_val_trans)

    # Validate Probabilities
    assert val_probs.shape == (
        len(X_val),
        n_classes,
    ), "Probability matrix shape mismatch"
    # Row sums should be approx 1.0
    row_sums = np.sum(val_probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Predict Labels
    val_preds = clf.predict(X_val_trans)
    assert len(val_preds) == len(X_val)

    print("Model verification passed.")

    # 5. Full Pipeline Integration
    print("\n[Step 5] Testing Full Pipeline Integration...")

    # This function handles everything from loading to submission generation
    # We use the same debug size.
    model.run_training_and_inference(debug_size=DEBUG_SIZE, load_cached_data=True)

    # Verify Submission File
    submission_path = config.OUTPUT_SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Check Submission Format
    # Rows: DEBUG_SIZE
    # Columns: 'id' + 99 species
    assert df_sub.shape[0] == DEBUG_SIZE, f"Submission should have {DEBUG_SIZE} rows"

    # Check 'id' column exists
    assert "id" in df_sub.columns, "Submission missing 'id' column"

    # Check that probabilities are within [0, 1]
    # Exclude 'id' column
    probs_only = df_sub.drop(columns=["id"])
    assert (probs_only.values >= 0).all() and (
        probs_only.values <= 1
    ).all(), "Probabilities out of bounds"

    print("Full pipeline verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
