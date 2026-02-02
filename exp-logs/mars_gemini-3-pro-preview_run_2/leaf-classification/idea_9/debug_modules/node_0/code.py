import os
import sys
import warnings
import numpy as np
import pandas as pd

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")

# Import provided library modules
from library.data_loader import load_dataset
from library.ensemble_engine import StackingEnsemble
from library.utils import save_submission
from library.model_factory import (
    get_linear_expert,
    get_generative_expert,
    get_kernel_expert,
)

# Set random seed for reproducibility in main script
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def validate_shapes(array, expected_rows, name):
    """Helper to validate array shapes."""
    assert (
        array.shape[0] == expected_rows
    ), f"{name} has {array.shape[0]} rows, expected {expected_rows}"
    print(f"Verified {name} shape: {array.shape}")


def main():
    print("Starting Leaf Classification Demo...")
    print("=" * 40)

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading Dataset...")
    # We force reload from metadata to demonstrate the full pipeline
    X_train, y_train, X_val, y_val, X_test, test_ids, class_names = load_dataset(
        load_cached_data=False
    )

    # Validate loaded data
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]
    n_features = X_train.shape[1]
    n_classes = len(class_names)

    print(f"Data Loaded:")
    print(f"  - Train samples: {n_train}")
    print(f"  - Val samples:   {n_val}")
    print(f"  - Test samples:  {n_test}")
    print(f"  - Features:      {n_features}")
    print(f"  - Classes:       {n_classes}")

    validate_shapes(X_train, n_train, "X_train")
    validate_shapes(X_val, n_val, "X_val")
    validate_shapes(X_test, n_test, "X_test")

    # Check that class names match the expected count
    assert len(class_names) == 99, f"Expected 99 classes, found {len(class_names)}"

    # -------------------------------------------------------------------------
    # 2. Model Factory Demonstration (Unit Test)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Model Factory...")
    # Quickly instantiate models to ensure factory works
    linear_model = get_linear_expert(RANDOM_SEED)
    gen_model = get_generative_expert(RANDOM_SEED)
    kernel_model = get_kernel_expert(RANDOM_SEED)

    print("Successfully instantiated Linear, Generative, and Kernel experts.")

    # -------------------------------------------------------------------------
    # 3. Stacking Ensemble Workflow
    # -------------------------------------------------------------------------
    print("\n[Step 3] Initializing Stacking Ensemble...")
    ensemble = StackingEnsemble(random_state=RANDOM_SEED)

    # A. Generate Out-of-Fold (OOF) Predictions
    # The ensemble uses 3-fold CV internally to generate predictions for the meta-learner
    print("Generating OOF predictions (this may take a minute)...")
    oof_preds = ensemble.generate_oof_predictions(X_train, y_train)

    # Validate OOF shape: (n_samples, n_models * n_classes)
    # n_models = 3 (Linear, Generative, Kernel)
    expected_cols = 3 * n_classes
    assert oof_preds.shape == (
        n_train,
        expected_cols,
    ), f"OOF shape mismatch. Got {oof_preds.shape}, expected {(n_train, expected_cols)}"
    print("OOF predictions generated successfully.")

    # B. Train Meta-Learner
    # The meta-learner learns to combine the 3 models based on OOF predictions
    print("Training Meta-Learner...")
    ensemble.train_meta_learner(oof_preds, y_train)

    # C. Retrain Base Models on Full Data
    # For best performance, we combine Train and Validation sets for the final fit
    print("Preparing full dataset (Train + Val) for final retraining...")
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    validate_shapes(X_full, n_train + n_val, "X_full")

    print("Retraining base models on full dataset...")
    ensemble.train_full_base_models(X_full, y_full)

    # -------------------------------------------------------------------------
    # 4. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[Step 4] Generating Test Predictions...")
    final_probs = ensemble.predict(X_test)

    # Validate prediction shape and values
    assert final_probs.shape == (
        n_test,
        n_classes,
    ), f"Prediction shape mismatch. Got {final_probs.shape}, expected {(n_test, n_classes)}"

    # Check probability constraints
    assert np.all(final_probs >= 0) and np.all(
        final_probs <= 1
    ), "Predictions contain values outside [0, 1]"

    # Check if rows sum roughly to 1 (floating point tolerance)
    row_sums = np.sum(final_probs, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    print("Predictions generated successfully.")

    # Save Submission
    print("Saving submission file...")
    submission_path = "./working/demo_submission/submission.csv"
    save_submission(final_probs, test_ids, class_names, output_path=submission_path)

    # Verify file creation
    if os.path.exists(submission_path):
        print(f"File created at: {submission_path}")
        # Load and check header
        df_sub = pd.read_csv(submission_path)
        print(f"Submission dimensions: {df_sub.shape}")

        # Check first few columns
        expected_cols = ["id"] + list(class_names)
        assert (
            list(df_sub.columns) == expected_cols
        ), "Submission columns do not match class names"
        assert (
            df_sub.shape[0] == n_test
        ), f"Submission has {df_sub.shape[0]} rows, expected {n_test}"
    else:
        raise FileNotFoundError("Submission file was not saved correctly.")

    print("\n" + "=" * 40)
    print("Demo execution completed successfully!")


if __name__ == "__main__":
    main()
