import os
import numpy as np
import pandas as pd
import warnings
import shutil

# Import provided library modules
from library import config
from library import data_processing
from library import model
from library import training


def main():
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("=== Setting up environment ===")
    warnings.filterwarnings("ignore")
    np.random.seed(config.SEED)

    # Ensure working directory exists (handled by config, but good to double check)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 2. Demonstrate and Validate Data Processing
    # -------------------------------------------------------------------------
    print("\n=== Testing Data Processing Module ===")

    # Force reload to demonstrate processing logic (load_cached_data=False)
    # This reads metadata, computes features, and saves to ./working
    (
        X_train,
        y_train,
        genus_train,
        X_val,
        y_val,
        genus_val,
        X_test,
        ids_test,
        classes,
    ) = data_processing.process_data(load_cached_data=False)

    # Assertions to verify data integrity
    print("Verifying data shapes and types...")
    assert (
        X_train.shape[0] == y_train.shape[0] == genus_train.shape[0]
    ), "Mismatch in training data dimensions"
    assert (
        X_val.shape[0] == y_val.shape[0] == genus_val.shape[0]
    ), "Mismatch in validation data dimensions"
    assert X_test.shape[0] == ids_test.shape[0], "Mismatch in test data dimensions"
    assert X_train.shape[1] == 192, "Incorrect feature count (expected 192)"
    assert (
        len(classes) == 99
    ), f"Incorrect class count (expected 99, got {len(classes)})"

    # Verify Genus extraction logic
    # Genus should be the prefix of Species (e.g., 'Acer' from 'Acer_Rubrum')
    sample_idx = 0
    sample_species = y_train[sample_idx]
    sample_genus = genus_train[sample_idx]
    assert sample_species.startswith(
        sample_genus
    ), f"Genus extraction failed: {sample_genus} not in {sample_species}"

    print("Data processing validation passed.")

    # 3. Demonstrate and Validate Model Logic
    # -------------------------------------------------------------------------
    print("\n=== Testing Model Logic (TaxonomicDualCentroidOAS) ===")

    # Instantiate the custom estimator
    # We use a lambda of 0.5 to test the mixing of species and genus means
    clf = model.TaxonomicDualCentroidOAS(lambda_reg=0.5)

    # Fit on a small subset to ensure logic holds
    subset_size = 100
    X_sub = X_train[:subset_size]
    y_sub = y_train[:subset_size]
    genus_sub = genus_train[:subset_size]

    clf.fit(X_sub, y_sub, genus_sub)

    # Check attributes were set
    assert hasattr(clf, "W_"), "Model W_ attribute missing after fit"
    assert hasattr(clf, "b_"), "Model b_ attribute missing after fit"
    assert hasattr(clf, "classes_"), "Model classes_ attribute missing after fit"

    # Predict on validation set
    print("Verifying prediction output...")
    probs = clf.predict_proba(X_val)
    preds = clf.predict(X_val)

    # Validate Probabilities
    # 1. Shape must be (n_samples, n_classes)
    assert probs.shape == (
        X_val.shape[0],
        len(classes),
    ), f"Probability shape mismatch: {probs.shape}"

    # 2. Rows must sum to 1 (within float tolerance)
    row_sums = probs.sum(axis=1)
    assert np.allclose(
        row_sums, 1.0
    ), f"Probabilities do not sum to 1. Max deviation: {np.max(np.abs(row_sums - 1.0))}"

    # 3. Values must be in [0, 1]
    assert probs.min() >= 0 and probs.max() <= 1, "Probabilities out of range [0, 1]"

    print("Model logic validation passed.")

    # 4. Demonstrate Full Training Pipeline
    # -------------------------------------------------------------------------
    print("\n=== Running Full Training Pipeline ===")

    # We use the provided training module to run the full workflow.
    # We reduce n_splits to 3 to optimize speed for this demonstration,
    # while still exercising the cross-validation logic.
    training.run_training(n_splits=3)

    # 5. Verify Submission Artifact
    # -------------------------------------------------------------------------
    print("\n=== Verifying Submission File ===")

    submission_path = config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not created at {submission_path}")

    df_sub = pd.read_csv(submission_path)

    # Check dimensions: Test samples x (ID + 99 Classes)
    expected_rows = len(ids_test)
    expected_cols = 1 + len(classes)  # 'id' + 99 species

    assert df_sub.shape == (
        expected_rows,
        expected_cols,
    ), f"Submission shape mismatch. Expected ({expected_rows}, {expected_cols}), got {df_sub.shape}"

    # Check ID column
    assert config.ID_COL in df_sub.columns, "ID column missing in submission"
    assert np.all(
        df_sub[config.ID_COL].values == ids_test
    ), "Submission IDs do not match test set IDs"

    # Check Class columns
    missing_cols = [c for c in classes if c not in df_sub.columns]
    assert not missing_cols, f"Missing class columns in submission: {missing_cols[:5]}"

    print("Submission file validation passed.")
    print(f"File location: {submission_path}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
