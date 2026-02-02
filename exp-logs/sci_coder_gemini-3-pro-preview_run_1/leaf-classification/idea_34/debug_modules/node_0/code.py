import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import log_loss

# Import provided library modules
import library.config as config
import library.data_loader as data_loader
import library.preprocessing as preprocessing
import library.model as model

# -----------------------------------------------------------------------------
# Setup & Configuration
# -----------------------------------------------------------------------------


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_demo():
    print("=== Starting Demonstration of Library Components ===")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    set_seed(config.SEED)

    # -------------------------------------------------------------------------
    # 1. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[1] Testing Data Loader...")

    # Force reload from metadata to verify raw loading logic
    # Note: We pass load_cached_data=False to ensure we test the processing logic
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = data_loader.load_data(
        load_cached_data=False
    )

    print(f"    Training Data Shape: {X_train.shape}")
    print(f"    Validation Data Shape: {X_val.shape}")
    print(f"    Test Data Shape: {X_test.shape}")
    print(f"    Number of Classes: {len(classes)}")

    # Assertions
    assert len(X_train) == len(y_train), "Mismatch in training samples and labels"
    assert len(X_val) == len(y_val), "Mismatch in validation samples and labels"
    assert len(X_test) == len(test_ids), "Mismatch in test samples and IDs"
    assert X_train.dtypes.apply(
        lambda x: x == np.float64
    ).all(), "X_train must be float64"
    assert isinstance(classes, np.ndarray), "Classes should be a numpy array"

    print("    -> Data Loader assertions passed.")

    # -------------------------------------------------------------------------
    # 2. Preprocessing Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Testing Preprocessing Pipeline...")

    # Apply preprocessing (Yeo-Johnson + StandardScaler)
    X_train_trans, X_val_trans, X_test_trans = preprocessing.preprocess_data(
        X_train, X_val, X_test, load_cached_data=False
    )

    print(f"    Transformed Train Shape: {X_train_trans.shape}")

    # Assertions
    assert isinstance(
        X_train_trans, np.ndarray
    ), "Transformed data should be numpy array"
    assert X_train_trans.shape == X_train.shape, "Shape mismatch after transformation"
    assert X_train_trans.dtype == np.float64, "Transformed data must be float64"

    # Check standardization properties on training set (approx mean 0, std 1)
    # Note: PowerTransformer might shift things slightly, but StandardScaler is last.
    train_mean = np.mean(X_train_trans, axis=0)
    train_std = np.std(X_train_trans, axis=0)

    assert np.allclose(
        train_mean, 0, atol=1e-6
    ), "Transformed training mean should be approx 0"
    assert np.allclose(
        train_std, 1, atol=1e-6
    ), "Transformed training std should be approx 1"

    print("    -> Preprocessing assertions passed.")

    # -------------------------------------------------------------------------
    # 3. Model Logic Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Testing CholeskyOASDiscriminant Model...")

    clf = model.CholeskyOASDiscriminant()

    # Fit model
    print("    Fitting model...")
    clf.fit(X_train_trans, y_train)

    # Check attributes
    assert hasattr(clf, "means_"), "Model missing means_ attribute"
    assert hasattr(clf, "covariance_"), "Model missing covariance_ attribute"
    assert hasattr(clf, "coef_"), "Model missing coef_ attribute"
    assert clf.coef_.shape == (
        len(classes),
        X_train.shape[1],
    ), "Coefficient matrix shape mismatch"

    # Predict probabilities on validation set
    print("    Predicting probabilities on validation set...")
    val_probs = clf.predict_proba(X_val_trans)

    # Verify probabilities
    assert val_probs.shape == (
        len(X_val),
        len(classes),
    ), "Probability output shape mismatch"
    row_sums = val_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"
    assert (val_probs >= 0).all() and (
        val_probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    # Calculate Log Loss
    # Need to encode y_val using the model's encoder to match column indices
    y_val_enc = clf.le_.transform(y_val)
    loss = log_loss(y_val_enc, val_probs, labels=range(len(classes)))
    print(f"    Validation Log Loss: {loss:.5f}")

    assert not np.isnan(loss), "Log loss is NaN"
    assert not np.isinf(loss), "Log loss is Inf"

    print("    -> Model assertions passed.")

    # -------------------------------------------------------------------------
    # 4. Full Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[4] Testing Full Pipeline (train_and_predict)...")

    # This function handles loading, preprocessing, training, and submission generation
    model.train_and_predict()

    # Verify submission file creation
    assert os.path.exists(config.SUBMISSION_FILE), "Submission file was not created"

    print("    -> Pipeline execution completed.")

    # -------------------------------------------------------------------------
    # 5. Submission File Validation
    # -------------------------------------------------------------------------
    print("\n[5] Validating Submission File...")

    df_sub = pd.read_csv(config.SUBMISSION_FILE)

    print(f"    Submission Shape: {df_sub.shape}")

    # Check columns
    expected_cols = ["id"] + list(classes)
    # Sort both to ensure matching sets, though order in file might differ slightly
    assert sorted(df_sub.columns.tolist()) == sorted(
        expected_cols
    ), "Submission columns mismatch"

    # Check row count
    assert len(df_sub) == len(
        test_ids
    ), f"Submission row count {len(df_sub)} != test set size {len(test_ids)}"

    # Check IDs
    assert set(df_sub["id"]) == set(test_ids), "Submission IDs do not match test IDs"

    # Check value ranges
    prob_cols = [c for c in df_sub.columns if c != "id"]
    probs = df_sub[prob_cols].values
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Submission contains probabilities outside [0, 1]"

    print("    -> Submission file is valid.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
