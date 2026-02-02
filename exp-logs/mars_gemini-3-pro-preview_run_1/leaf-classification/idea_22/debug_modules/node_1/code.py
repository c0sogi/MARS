import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import warnings

# Import library modules
from library.math_utils import compute_geometric_median
from library.data_manager import load_dataset
from library.robust_classifier import GeometricOASDiscriminant, run_training_pipeline
from library.config import WORKING_DIR, OUTPUT_DIR

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def demonstrate_math_utils():
    print("\n=== Demonstrating library.math_utils ===")

    # 1. Create a synthetic cluster of points around (0,0)
    # Shape: (100, 2)
    X = np.random.randn(100, 2)
    true_center = np.mean(X, axis=0)

    # 2. Add a massive outlier
    outlier = np.array([[1000.0, 1000.0]])
    X_outlier = np.vstack([X, outlier])

    # 3. Compute Arithmetic Mean (sensitive to outliers)
    arithmetic_mean = np.mean(X_outlier, axis=0)

    # 4. Compute Geometric Median (robust to outliers)
    geo_median = compute_geometric_median(X_outlier, eps=1e-6, max_iter=100)

    print(f"True Center (no outlier): {true_center}")
    print(f"Arithmetic Mean (with outlier): {arithmetic_mean}")
    print(f"Geometric Median (with outlier): {geo_median}")

    # 5. Validation Logic
    # The geometric median should be much closer to the true center than the arithmetic mean
    dist_mean = np.linalg.norm(arithmetic_mean - true_center)
    dist_geo = np.linalg.norm(geo_median - true_center)

    print(f"Shift caused by outlier - Mean: {dist_mean:.4f}, GeoMedian: {dist_geo:.4f}")

    assert (
        dist_geo < dist_mean
    ), "Geometric Median should be more robust to outliers than Arithmetic Mean"
    assert dist_geo < 1.0, "Geometric Median shifted too far from true center"
    print("Assertion Passed: Geometric Median is robust.")


def demonstrate_data_manager():
    print("\n=== Demonstrating library.data_manager ===")

    # Force reload from metadata to test preprocessing pipeline
    # This applies Yeo-Johnson + StandardScaler
    print("Loading dataset with Inductive Preprocessing...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_dataset(
        load_cached_data=False
    )

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")
    print(f"Number of Classes: {len(classes)}")

    # Validation Logic
    # 1. Check Shapes
    assert X_train.shape[1] == 192, "Expected 192 features (margin + shape + texture)"
    assert len(classes) == 99, "Expected 99 plant species"

    # 2. Check Preprocessing Statistics (StandardScaler should result in mean~0, std~1)
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)

    # We check the average of means/stds across features to ensure global scaling worked
    avg_mean = np.mean(np.abs(train_mean))
    avg_std = np.mean(train_std)

    print(f"Avg Feature Mean (should be ~0): {avg_mean:.6f}")
    print(f"Avg Feature Std (should be ~1): {avg_std:.6f}")

    assert (
        avg_mean < 1e-2
    ), "Features do not appear to be centered (StandardScaler failed?)"
    assert 0.9 < avg_std < 1.1, "Features do not appear to be scaled to unit variance"

    print("Assertion Passed: Data loaded and preprocessed correctly.")
    return X_train, y_train, X_val, y_val, classes


def demonstrate_robust_classifier(X_train, y_train, X_val, y_val, classes):
    print("\n=== Demonstrating library.robust_classifier ===")

    # Instantiate the custom classifier
    clf = GeometricOASDiscriminant()

    # Fit
    print("Fitting GeometricOASDiscriminant...")
    clf.fit(X_train, y_train)

    # Validate Attributes
    assert hasattr(clf, "centroids_"), "Model failed to compute centroids"
    assert hasattr(clf, "covariance_"), "Model failed to compute covariance"
    assert clf.centroids_.shape == (99, 192), "Centroids shape mismatch"

    # Predict
    print("Predicting probabilities on validation set...")
    probs = clf.predict_proba(X_val)

    # Validation Logic
    # 1. Check Probability Properties
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"
    assert probs.min() >= 0 and probs.max() <= 1, "Probabilities out of range [0, 1]"

    # 2. Check Metric
    loss = log_loss(y_val, probs, labels=range(len(classes)))
    print(f"Validation Log Loss: {loss:.4f}")

    # Basic sanity check: Random guessing for 99 classes is -ln(1/99) ~= 4.6
    # A trained model should be significantly better
    assert (
        loss < 4.0
    ), f"Model performance ({loss}) is worse than random guessing (~4.6)"

    print("Assertion Passed: Classifier fits and predicts valid probabilities.")


def demonstrate_full_pipeline():
    print("\n=== Demonstrating Full Pipeline Execution ===")
    # This runs the `run_training_pipeline` function which orchestrates everything
    # and saves the submission file.
    run_training_pipeline(load_cached=True)

    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission File Loaded: {df_sub.shape}")
    assert df_sub.shape[0] == 99, "Submission should have 99 rows (test set size)"
    # 100 columns = 1 id + 99 classes
    assert df_sub.shape[1] == 100, "Submission should have 100 columns"

    print("Assertion Passed: Full pipeline executed successfully.")


if __name__ == "__main__":
    set_seed(42)

    # 1. Test Math Utils
    demonstrate_math_utils()

    # 2. Test Data Manager
    X_train, y_train, X_val, y_val, classes = demonstrate_data_manager()

    # 3. Test Robust Classifier
    demonstrate_robust_classifier(X_train, y_train, X_val, y_val, classes)

    # 4. Run End-to-End Pipeline
    demonstrate_full_pipeline()

    print("\nAll demonstrations and validations completed successfully.")
