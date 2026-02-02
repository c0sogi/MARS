import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import provided library components
from library.config import RANDOM_SEED, SUBMISSION_PATH, WORKING_DIR, FLOAT_PRECISION
from library.transformations import get_transformed_data
from library.oas_model import OASLinearDiscriminant, create_submission_file


def run_demo():
    print("Initializing Plant Species Classification Pipeline...")

    # 1. Set Random Seed for Reproducibility
    np.random.seed(RANDOM_SEED)

    # 2. Execute Data Pipeline (Loading + Feature Extraction + Transformation)
    # We set load_cached_data=False to demonstrate the full execution flow
    # including geometric feature extraction and iterative gaussianization.
    print("\n[Step 1] Running Data Transformation Pipeline...")
    ((X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, ids_test)) = (
        get_transformed_data(load_cached_data=False)
    )

    # Validation: Check Data Integrity
    print(f"  Training Data Shape: {X_train.shape}")
    print(f"  Validation Data Shape: {X_val.shape}")
    print(f"  Test Data Shape: {X_test.shape}")

    assert X_train.shape[0] > 0, "Training set is empty"
    assert X_val.shape[0] > 0, "Validation set is empty"
    assert X_test.shape[0] > 0, "Test set is empty"
    assert not np.isnan(X_train).any(), "NaN values found in training data"
    assert not np.isnan(X_val).any(), "NaN values found in validation data"
    assert not np.isnan(X_test).any(), "NaN values found in test data"

    # 3. Model Training (OAS Linear Discriminant)
    print("\n[Step 2] Training OAS Linear Discriminant Model...")
    model = OASLinearDiscriminant()
    model.fit(X_train, y_train)

    # Validation: Check Model State
    assert model.W_ is not None, "Model weights (W) not initialized"
    assert model.b_ is not None, "Model bias (b) not initialized"

    # 4. Model Evaluation
    print("\n[Step 3] Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss
    # We need to encode y_val to integers or use the string labels if log_loss supports it.
    # Scikit-learn log_loss handles string labels if provided in 'labels' parameter.
    loss = log_loss(y_val, val_probs, labels=model.classes_)
    print(f"  Validation Log Loss: {loss:.4f}")

    # Validation: Performance and Probability Check
    # A random guess on 99 classes would be ln(99) ~= 4.6.
    # A good model should be significantly lower.
    assert loss < 2.0, f"Model performance is poor (Log Loss: {loss:.4f})"

    # Check probability sums
    row_sums = np.sum(val_probs, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities do not sum to 1"

    # 5. Generate Submission
    print("\n[Step 4] Generating Submission File...")
    create_submission_file(model, X_test, ids_test, SUBMISSION_PATH)

    # Validation: Submission File Integrity
    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created"

    df_submission = pd.read_csv(SUBMISSION_PATH)
    print(f"  Submission loaded. Shape: {df_submission.shape}")

    # Check Dimensions: 99 test samples + header
    expected_rows = len(ids_test)
    # Check Columns: id + 99 classes
    expected_cols = 1 + len(model.classes_)

    assert (
        df_submission.shape[0] == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {df_submission.shape[0]}"
    assert (
        df_submission.shape[1] == expected_cols
    ), f"Submission column count mismatch. Expected {expected_cols}, got {df_submission.shape[1]}"

    # Check IDs match
    submission_ids = df_submission["id"].values
    # Sort both to ensure set equality check works regardless of order (though order usually preserved)
    assert np.array_equal(
        np.sort(submission_ids), np.sort(ids_test)
    ), "Submission IDs do not match Test IDs"

    print("\n[Success] Pipeline completed successfully. Submission is ready.")


if __name__ == "__main__":
    run_demo()
