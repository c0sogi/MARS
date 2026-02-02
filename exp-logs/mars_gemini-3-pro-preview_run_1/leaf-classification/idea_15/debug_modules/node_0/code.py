import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import WORKING_DIR, SUBMISSION_PATH, FEATURE_COLS, TARGET_COL, SEED
from library.data_loader import load_dataset
from library.preprocessing import GaussianTransformer, get_preprocessed_data
from library.oas_discriminant import OASLinearDiscriminant, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("--- Starting Library Demonstration Script ---")
    set_seed(SEED)

    # ==========================================
    # 1. Demonstrate Data Loading
    # ==========================================
    print("\n[1] Demonstrating Data Loader...")

    # Force load from metadata to verify raw loading logic
    X_train_raw, y_train, ids_train, X_val_raw, y_val, ids_val, X_test_raw, ids_test = (
        load_dataset(load_cached_data=False)
    )

    # Verify shapes
    n_features = len(FEATURE_COLS)
    print(f"    Training Data Shape: {X_train_raw.shape}")
    print(f"    Validation Data Shape: {X_val_raw.shape}")

    # Assertions to ensure data integrity
    assert (
        X_train_raw.shape[1] == n_features
    ), f"Expected {n_features} features, got {X_train_raw.shape[1]}"
    assert len(y_train) == len(
        X_train_raw
    ), "Mismatch between X_train and y_train length"
    assert len(ids_train) == len(
        X_train_raw
    ), "Mismatch between X_train and ids_train length"
    assert not X_train_raw.isnull().values.any(), "Raw training data contains NaNs"

    print("    Data loading verification passed.")

    # ==========================================
    # 2. Demonstrate Preprocessing (GaussianTransformer)
    # ==========================================
    print("\n[2] Demonstrating GaussianTransformer...")

    # Instantiate the transformer
    transformer = GaussianTransformer()

    # Fit on training data
    # Using a subset for demonstration speed if dataset were huge, but here it's small (~700 rows)
    transformer.fit(X_train_raw)

    # Transform validation data
    X_val_trans = transformer.transform(X_val_raw)

    # Verify statistics of transformed data (Should be close to Mean=0, Std=1)
    val_means = np.mean(X_val_trans, axis=0)
    val_stds = np.std(X_val_trans, axis=0)

    # We allow some deviation because we fit on Train and transformed Val
    print(f"    Transformed Val Mean (avg across features): {np.mean(val_means):.4f}")
    print(f"    Transformed Val Std  (avg across features): {np.mean(val_stds):.4f}")

    assert (
        X_val_trans.shape == X_val_raw.shape
    ), "Transformer changed data shape unexpectedly"
    assert (
        X_val_trans.dtype == np.float64
    ), "Transformer did not enforce float64 precision"

    # Use the high-level function to get cached/processed data for the next steps
    print("    Calling get_preprocessed_data() to retrieve full processed sets...")
    (X_train_p, y_train_p, _, X_val_p, y_val_p, _, X_test_p, _) = get_preprocessed_data(
        load_cached_data=True
    )

    assert np.allclose(
        X_train_p.mean(), 0, atol=0.1
    ), "Cached processed data does not appear centered"
    print("    Preprocessing verification passed.")

    # ==========================================
    # 3. Demonstrate OAS Linear Discriminant Model
    # ==========================================
    print("\n[3] Demonstrating OASLinearDiscriminant...")

    model = OASLinearDiscriminant()

    # Fit the model
    model.fit(X_train_p, y_train_p)

    # Check internal attributes
    n_classes = len(np.unique(y_train_p))
    assert (
        len(model.classes_) == n_classes
    ), "Model identified incorrect number of classes"
    assert model.means_.shape == (n_classes, n_features), "Model means shape incorrect"

    # Predict probabilities on validation set
    val_probs = model.predict_proba(X_val_p)

    # Verify probabilities
    # 1. Sum to 1
    row_sums = np.sum(val_probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # 2. Range [0, 1]
    assert (val_probs >= 0).all() and (
        val_probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    # 3. Shape
    assert val_probs.shape == (
        len(y_val_p),
        n_classes,
    ), "Probability matrix shape incorrect"

    # Calculate simple accuracy for demonstration
    val_preds = model.classes_[np.argmax(val_probs, axis=1)]
    acc = np.mean(val_preds == y_val_p)
    print(f"    Validation Accuracy: {acc:.4f}")

    print("    Model verification passed.")

    # ==========================================
    # 4. Demonstrate Full Pipeline (Submission Generation)
    # ==========================================
    print("\n[4] Executing Full Submission Pipeline...")

    # This function handles retraining on Train+Val and predicting on Test
    generate_submission()

    # Verify the output file
    if os.path.exists(SUBMISSION_PATH):
        df_sub = pd.read_csv(SUBMISSION_PATH)
        print(f"    Submission file generated at: {SUBMISSION_PATH}")
        print(f"    Submission Shape: {df_sub.shape}")

        # Check columns
        expected_cols = ["id"] + sorted(list(np.unique(y_train_p)))
        # Note: The model classes might be sorted differently depending on implementation,
        # but OASLinearDiscriminant uses np.unique which sorts alphanumerically.
        # We check if 'id' is present and we have the correct number of columns.
        assert "id" in df_sub.columns, "Submission missing 'id' column"
        assert (
            df_sub.shape[1] == n_classes + 1
        ), f"Expected {n_classes + 1} columns, found {df_sub.shape[1]}"

        # Check ID count matches test set
        assert len(df_sub) == len(
            ids_test
        ), f"Expected {len(ids_test)} rows, found {len(df_sub)}"

        print("    Submission file format verification passed.")
    else:
        raise FileNotFoundError(f"Submission file was not created at {SUBMISSION_PATH}")

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    main()
