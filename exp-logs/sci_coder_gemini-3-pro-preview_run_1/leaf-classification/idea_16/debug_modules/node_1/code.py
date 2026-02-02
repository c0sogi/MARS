import os
import numpy as np
import pandas as pd
import shutil
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import preprocessing
from library import model
from library import solver


def test_utils():
    print("\n=== Testing Utils ===")

    # Test 1: normalize_and_clip_probabilities
    # Create raw predictions that don't sum to 1
    raw_preds = np.array(
        [
            [0.2, 0.2],  # Sums to 0.4 -> should become [0.5, 0.5]
            [10.0, 0.0],  # Sums to 10.0 -> should become [1.0, 0.0]
            [0.0, 0.0],  # Sums to 0.0 -> handled safely, usually uniform or preserved
        ]
    )

    processed = utils.normalize_and_clip_probabilities(raw_preds)

    # Check sums
    row_sums = processed.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Rows should sum to 1, got {row_sums}"

    # Check clipping
    assert (
        processed.min() >= config.PROB_CLIP_MIN
    ), "Probabilities below min clip threshold"
    assert (
        processed.max() <= config.PROB_CLIP_MAX
    ), "Probabilities above max clip threshold"

    print("Utils: Normalization and clipping verified.")

    # Test 2: compute_log_loss
    y_true = np.array([0, 1])
    # Perfect predictions (clipped)
    y_pred_good = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Bad predictions
    y_pred_bad = np.array([[0.0, 1.0], [1.0, 0.0]])

    loss_good = utils.compute_log_loss(y_true, y_pred_good, labels=[0, 1])
    loss_bad = utils.compute_log_loss(y_true, y_pred_bad, labels=[0, 1])

    assert loss_good < 0.01, f"Expected low loss for good preds, got {loss_good}"
    assert loss_bad > 10.0, f"Expected high loss for bad preds, got {loss_bad}"

    print("Utils: Log loss computation verified.")


def test_model_logic():
    print("\n=== Testing FixedMeanOASDiscriminant (Unit Test) ===")

    # Generate synthetic data: 2 classes, 10 features, separated
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=config.RANDOM_SEED,
        weights=[0.5, 0.5],
    )

    # Split
    split = 150
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    clf = model.FixedMeanOASDiscriminant()

    # 1. Fit Means
    clf.fit_means(X_train, y_train)
    assert clf.means_.shape == (2, 10), "Means shape mismatch"
    assert clf.priors_.shape == (2,), "Priors shape mismatch"

    # 2. Compute Residuals
    residuals = clf.compute_residuals(X_train, y_train)
    assert residuals.shape == X_train.shape, "Residuals shape mismatch"
    # Residuals should have mean close to 0
    assert np.abs(residuals.mean()) < 0.1, "Residuals not centered"

    # 3. Fit Covariance
    clf.fit_covariance(residuals)
    assert clf.precision_.shape == (10, 10), "Precision matrix shape mismatch"
    assert clf.coef_ is not None, "Coefficients not calculated"

    # 4. Predict
    probs = clf.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2), "Prediction shape mismatch"
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities must sum to 1"

    # Simple accuracy check (should be high for this synthetic dataset)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)
    print(f"Model Unit Test Accuracy: {acc:.4f}")
    assert acc > 0.8, "Model accuracy unexpectedly low on synthetic data"

    print("Model: Logic verified.")


def test_data_and_preprocessing():
    print("\n=== Testing Data Loading and Preprocessing ===")

    # Force reload from source to verify raw loading logic
    # We use a small subset logic implicitly by relying on the provided small dataset

    # 1. Load Raw
    X_train, y_train, X_val, y_val, X_test, test_ids = data_loader.load_datasets(
        load_cached_data=False
    )

    print(f"Raw Train Shape: {X_train.shape}")
    print(f"Raw Val Shape: {X_val.shape}")
    print(f"Raw Test Shape: {X_test.shape}")

    assert not X_train.isnull().values.any(), "NaNs found in raw training data"
    assert len(y_train) == len(X_train), "Mismatch in X_train and y_train length"

    # 2. Preprocessing
    # We test the RobustPreprocessor class directly
    preprocessor = preprocessing.RobustPreprocessor()

    # Fit on train
    preprocessor.fit(X_train)
    assert preprocessor.is_fitted, "Preprocessor should be fitted"

    # Transform
    X_train_trans = preprocessor.transform(X_train)

    # Check stats (StandardScaler should make mean ~0 and std ~1)
    # Note: PowerTransformer is applied first, so it might not be exactly 0/1 immediately
    # if PT doesn't perfectly Gaussianize, but Scaler is applied last.
    means = np.mean(X_train_trans, axis=0)
    stds = np.std(X_train_trans, axis=0)

    # Allow some floating point tolerance
    assert np.all(np.abs(means) < 1e-5), "Transformed features should have mean ~0"
    assert np.all(np.abs(stds - 1.0) < 1e-5), "Transformed features should have std ~1"

    print("Preprocessing: Pipeline verified.")


def test_solver_integration():
    print("\n=== Testing Full Solver Integration ===")

    # Clean up any previous submission
    if os.path.exists(config.SUBMISSION_PATH):
        os.remove(config.SUBMISSION_PATH)

    # Run Solver (force reload to ensure end-to-end validity)
    # This runs the semi-supervised pipeline
    solver.run_solver(load_cached_data=False)

    # Verify Submission
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")

    # Check columns
    # 1 ID column + 99 Species columns = 100 columns
    assert df_sub.shape[1] == 100, f"Expected 100 columns, got {df_sub.shape[1]}"
    assert config.ID_COL in df_sub.columns, "ID column missing"

    # Check ID integrity
    test_df_orig = pd.read_csv(config.TEST_PATH)
    assert set(df_sub[config.ID_COL]) == set(
        test_df_orig[config.ID_COL]
    ), "Submission IDs do not match Test IDs"

    # Check probability range
    feature_cols = [c for c in df_sub.columns if c != config.ID_COL]
    probs = df_sub[feature_cols].values
    assert probs.min() >= 0, "Negative probabilities found"
    assert probs.max() <= 1, "Probabilities > 1 found"

    # Check row sums (should be approximately 1 due to normalization in solver/model)
    # The model output is softmax, so it sums to 1.
    # Note: The solver saves raw probabilities from predict_proba.
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Submission rows do not sum to 1"

    print("Solver: Integration verified successfully.")


if __name__ == "__main__":
    # Ensure reproducible runs
    utils.set_seed(config.RANDOM_SEED)

    print("Starting Demonstration Script...")

    try:
        test_utils()
        test_model_logic()
        test_data_and_preprocessing()
        test_solver_integration()

        print("\nAll tests passed successfully!")

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        exit(1)
