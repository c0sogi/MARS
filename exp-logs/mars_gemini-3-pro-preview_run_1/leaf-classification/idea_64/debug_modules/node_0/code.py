import sys
import os
import shutil
import numpy as np
import pandas as pd

# Ensure the current directory is in the python path to import from library
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import SEED, SUBMISSION_FILE, WORKING_DIR
from library.utils import set_seed, compute_log_loss
from library.data_loader import load_and_process_data
from library.preprocessing import preprocess_data
from library.model import OASDiscriminant, run_training_and_submission


def clean_working_directory():
    """
    Cleans the working directory to ensure the demonstration runs
    feature extraction and processing from scratch.
    """
    if os.path.exists(WORKING_DIR):
        try:
            shutil.rmtree(WORKING_DIR)
            print(f"Cleaned working directory: {WORKING_DIR}")
        except Exception as e:
            print(f"Warning: Could not clean working directory: {e}")
    os.makedirs(WORKING_DIR, exist_ok=True)


def main():
    print("============================================================")
    print("      Plant Species Identification Pipeline Demonstration   ")
    print("============================================================")

    # 1. Setup
    set_seed(SEED)
    clean_working_directory()

    # 2. Data Loading and Feature Extraction
    print("\n[Step 1] Loading Data and Extracting Features...")
    # We set load_cached_data=False to demonstrate the raw processing logic
    X_train, y_train, X_val, y_val, X_test, test_ids = load_and_process_data(
        load_cached_data=False
    )

    # Verification: Check Data Shapes
    # Expected features: 192 tabular + 6 geometric = 198
    EXPECTED_FEATURES = 198
    print(f"  Train Data Shape: {X_train.shape}")
    print(f"  Val Data Shape:   {X_val.shape}")
    print(f"  Test Data Shape:  {X_test.shape}")

    assert (
        X_train.shape[1] == EXPECTED_FEATURES
    ), f"Expected {EXPECTED_FEATURES} features, got {X_train.shape[1]}"
    assert len(y_train) == X_train.shape[0], "Mismatch between X_train and y_train"
    assert len(test_ids) == X_test.shape[0], "Mismatch between X_test and test_ids"
    print("  ✓ Data shapes verified.")

    # 3. Preprocessing
    print(
        "\n[Step 2] Preprocessing (VarianceThreshold -> PowerTransformer -> StandardScaler)..."
    )
    # Fits on Train, Transforms Train/Val/Test
    X_train_trans, X_val_trans, X_test_trans = preprocess_data(
        X_train, X_val, X_test, load_cached_data=False
    )

    # Verification: Check Statistics (StandardScaler should result in mean~0, std~1)
    train_mean = np.mean(X_train_trans, axis=0)
    train_std = np.std(X_train_trans, axis=0)

    print(f"  Transformed Train Mean (abs avg): {np.mean(np.abs(train_mean)):.6f}")
    print(f"  Transformed Train Std (avg):      {np.mean(train_std):.6f}")

    assert np.allclose(
        train_mean, 0, atol=1e-5
    ), "Transformed features are not centered."
    assert np.allclose(
        train_std, 1, atol=1e-5
    ), "Transformed features are not scaled to unit variance."
    print("  ✓ Preprocessing statistics verified.")

    # 4. Model Training
    print("\n[Step 3] Training OAS Discriminant Model...")
    model = OASDiscriminant()
    model.fit(X_train_trans, y_train)

    # Verification: Check Model Attributes
    assert hasattr(model, "coef_"), "Model missing 'coef_' attribute."
    assert hasattr(model, "intercept_"), "Model missing 'intercept_' attribute."
    assert hasattr(model, "classes_"), "Model missing 'classes_' attribute."

    n_classes = len(model.classes_)
    n_features = X_train_trans.shape[1]

    # Coef shape should be (n_classes, n_features) for LDA/QDA derived linear boundaries
    assert model.coef_.shape == (
        n_classes,
        n_features,
    ), f"Model coefficients shape mismatch. Expected {(n_classes, n_features)}, got {model.coef_.shape}"
    print(f"  Model trained on {n_classes} classes.")
    print("  ✓ Model structure verified.")

    # 5. Validation and Metric
    print("\n[Step 4] Validating and Computing Log Loss...")
    val_probs = model.predict_proba(X_val_trans)

    # Verification: Probabilities
    row_sums = np.sum(val_probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Predicted probabilities do not sum to 1."

    # Compute Metric
    loss = compute_log_loss(y_val, val_probs, model.classes_)
    print(f"  Validation Multi-class Log Loss: {loss:.5f}")
    assert loss > 0, "Log loss must be positive."
    print("  ✓ Metric calculation verified.")

    # 6. Full Pipeline Execution (Submission)
    print("\n[Step 5] Executing Full Pipeline Wrapper (Retraining on Full Data)...")
    # This function retrains on Train + Val and generates the submission file.
    # We allow it to load the cache we just created to save time.
    run_training_and_submission(load_cached_data=True)

    # Verification: Submission File
    if not os.path.exists(SUBMISSION_FILE):
        raise FileNotFoundError(f"Submission file not found at {SUBMISSION_FILE}")

    df_sub = pd.read_csv(SUBMISSION_FILE)
    print(f"  Submission File Shape: {df_sub.shape}")

    # Expected columns: 'id' + one column per class
    expected_cols = 1 + n_classes
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Submission has {df_sub.shape[1]} columns, expected {expected_cols}"
    assert df_sub.shape[0] == len(
        test_ids
    ), f"Submission has {df_sub.shape[0]} rows, expected {len(test_ids)}"

    # Check if probabilities are valid
    prob_cols = df_sub.columns[1:]
    probs = df_sub[prob_cols].values
    assert np.all(
        (probs >= 0) & (probs <= 1)
    ), "Submission contains probabilities out of [0, 1] range."

    print("  ✓ Submission file verified.")
    print("\n============================================================")
    print("           Demonstration Completed Successfully             ")
    print("============================================================")


if __name__ == "__main__":
    main()
