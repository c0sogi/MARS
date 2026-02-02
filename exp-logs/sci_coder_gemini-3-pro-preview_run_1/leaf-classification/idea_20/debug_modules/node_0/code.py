import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import functions and classes from the provided library
from library.config import SEED, WORKING_DIR
from library.utils import set_seed, save_submission
from library.data_loader import load_datasets
from library.preprocessing import get_transformed_data
from library.model import StabilizedOASDiscriminant


def main():
    print("--- Starting Library Usage Demo ---")

    # 1. Setup
    # Ensure reproducibility
    set_seed(SEED)
    print(f"Random seed set to {SEED}.")

    # 2. Data Loading
    # Load datasets (uses caching mechanism internally for speed)
    print("\n[1/5] Loading datasets...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_datasets(
        load_cached_data=True
    )

    # Verify Data Integrity
    n_features = 192
    n_classes = 99

    assert (
        X_train.shape[1] == n_features
    ), f"Expected {n_features} features, got {X_train.shape[1]}"
    assert len(X_train) == len(y_train), "Mismatch between training samples and labels."
    assert (
        len(classes) == n_classes
    ), f"Expected {n_classes} classes, got {len(classes)}."

    print(f"Data loaded successfully.")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples:   {len(X_val)}")
    print(f"  Test samples:  {len(X_test)}")

    # 3. Preprocessing
    # Apply Yeo-Johnson transformation and Standard Scaling
    print("\n[2/5] Preprocessing features...")
    X_train_trans, X_val_trans, X_test_trans = get_transformed_data(
        X_train, X_val, X_test, load_cached_data=True
    )

    # Verify Preprocessing Logic (StandardScaler should yield mean=0, std=1 on train)
    train_means = np.mean(X_train_trans, axis=0)
    train_stds = np.std(X_train_trans, axis=0)

    # Tolerances for floating point arithmetic
    assert np.all(
        np.abs(train_means) < 1e-5
    ), "Transformed training means should be approximately 0."
    assert np.all(
        np.abs(train_stds - 1.0) < 1e-5
    ), "Transformed training stds should be approximately 1."

    print("Preprocessing verification passed: Data is centered and scaled.")

    # 4. Model Training
    # Instantiate and fit the Stabilized OAS Discriminant model
    print("\n[3/5] Training StabilizedOASDiscriminant model...")
    model = StabilizedOASDiscriminant()
    model.fit(X_train_trans, y_train)

    # Verify Model State
    assert hasattr(model, "W_"), "Model should have weight matrix W_ after fitting."
    assert hasattr(model, "b_"), "Model should have bias vector b_ after fitting."
    assert model.W_.shape == (
        n_classes,
        n_features,
    ), f"Weight matrix shape mismatch. Expected ({n_classes}, {n_features}), got {model.W_.shape}"

    print("Model trained successfully.")

    # 5. Validation
    print("\n[4/5] Evaluating on validation set...")
    val_probs = model.predict_proba(X_val_trans)

    # Verify Probability Properties
    # Rows should sum to 1
    row_sums = np.sum(val_probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Predicted probabilities do not sum to 1."

    # Calculate Log Loss
    # Random guessing for 99 classes is -ln(1/99) ~= 4.595
    metric = log_loss(y_val, val_probs, labels=classes)
    print(f"Validation Log Loss: {metric:.6f}")

    # Assert model is learning (loss should be significantly lower than random guess)
    assert (
        metric < 4.0
    ), f"Model performance ({metric}) is worse than random guessing (~4.6)."

    # 6. Test Prediction & Submission
    print("\n[5/5] Generating submission...")
    test_probs = model.predict_proba(X_test_trans)

    # Define output path in working directory
    submission_path = os.path.join(WORKING_DIR, "demo_submission.csv")

    # Save submission
    save_submission(test_ids, test_probs, classes, submission_path)

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    # Shape should be (n_test_samples, 1 + n_classes) -> 1 for 'id' column
    expected_cols = n_classes + 1
    assert df_sub.shape == (
        len(test_ids),
        expected_cols,
    ), f"Submission shape mismatch. Expected ({len(test_ids)}, {expected_cols}), got {df_sub.shape}"

    print(f"Submission generated at: {submission_path}")
    print("--- Demo Completed Successfully ---")


if __name__ == "__main__":
    main()
