import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

# Import functions from the provided library
from library.utils import set_seed
from library.data_loader import load_full_dataset
from library.ensemble_trainer import train_ensemble
from library.inference import generate_submission, predict_ensemble


def main():
    print("Starting Leaf Classification Demonstration...")

    # 1. Setup and Reproducibility
    set_seed(42)

    # 2. Data Loading
    # We force load_cached_data=False to demonstrate the full loading pipeline from metadata CSVs
    print("\n[Step 1] Loading Data...")
    X_train, y_train, X_test, test_ids, classes = load_full_dataset(
        load_cached_data=False
    )

    # Validation of loaded data
    n_samples_train, n_features = X_train.shape
    n_classes = len(classes)

    print(f"  - Training Samples: {n_samples_train}")
    print(f"  - Features: {n_features}")
    print(f"  - Classes: {n_classes}")

    assert n_samples_train > 0, "Training set is empty."
    assert (
        n_features == 192
    ), f"Expected 192 features (64 margin + 64 shape + 64 texture), got {n_features}."
    assert len(X_test) == 99, f"Expected 99 test samples, got {len(X_test)}."

    # 3. Model Training
    # The train_ensemble function handles the complexity of building and fitting
    # the Linear (Logistic Regression), Generative (LDA), and Quadratic (PCA+Poly+LR) models.
    print("\n[Step 2] Training Ensemble Models...")
    models = train_ensemble(X_train, y_train, random_state=42)

    # Validate model dictionary
    expected_keys = ["linear", "generative", "quadratic"]
    for key in expected_keys:
        assert key in models, f"Model dictionary missing key: {key}"
        assert hasattr(
            models[key], "predict_proba"
        ), f"Model {key} does not have predict_proba method."

    print("  - All ensemble components trained successfully.")

    # 4. In-Memory Validation (Sanity Check)
    # We run a quick prediction on a small subset of training data to verify the pipeline mechanics
    print("\n[Step 3] Verifying Inference Logic on Training Subset...")
    subset_size = 10
    X_subset = X_train[:subset_size]
    y_subset = y_train[:subset_size]

    # Get raw probabilities from the ensemble
    probs_subset = predict_ensemble(models, X_subset)

    # Check shape
    assert probs_subset.shape == (
        subset_size,
        n_classes,
    ), f"Prediction shape mismatch. Expected {(subset_size, n_classes)}, got {probs_subset.shape}"

    # Check probability range
    assert np.all(probs_subset >= 0.0) and np.all(
        probs_subset <= 1.0
    ), "Probabilities out of range [0, 1]."

    # Calculate log loss on this small subset just to ensure metric calculation works
    subset_loss = log_loss(y_subset, probs_subset, labels=range(n_classes))
    print(f"  - Subset Log Loss: {subset_loss:.4f}")

    # 5. Generate Submission
    print("\n[Step 4] Generating Submission for Test Set...")
    output_file = "submission.csv"
    generate_submission(models, X_test, test_ids, classes, output_path=output_file)

    # 6. Final File Verification
    print("\n[Step 5] Verifying Submission File...")
    assert os.path.exists(output_file), "Submission file was not created."

    df_sub = pd.read_csv(output_file)

    # Check dimensions
    # Rows should be equal to test samples
    assert len(df_sub) == len(
        test_ids
    ), f"Submission row count mismatch. Expected {len(test_ids)}, got {len(df_sub)}."

    # Columns should be id + one per class
    expected_cols = 1 + n_classes
    assert (
        len(df_sub.columns) == expected_cols
    ), f"Submission column count mismatch. Expected {expected_cols}, got {len(df_sub.columns)}."

    # Check ID column
    assert "id" in df_sub.columns, "Submission missing 'id' column."
    assert np.all(
        df_sub["id"].values == test_ids
    ), "Submission IDs do not match test IDs."

    # Check values
    feature_cols = [c for c in df_sub.columns if c != "id"]
    values = df_sub[feature_cols].values
    assert np.all(values >= 0) and np.all(
        values <= 1
    ), "Submission contains values outside [0, 1]."

    print("  - Submission file passed all checks.")
    print("\nDemonstration complete successfully.")


if __name__ == "__main__":
    main()
