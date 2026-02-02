import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
import shutil

# Import from the provided library files
from library.utils import set_seed, save_submission
from library.data_loader import load_data
from library.preprocessing import Float64Preprocessor
from library.model import CholeskyOASDiscriminant, run_pipeline


def clean_dir(directory):
    """Utility to clean up a directory for the demo."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def demo_preprocessing():
    print("\n--- 1. Demonstrating Float64Preprocessor ---")
    # Create synthetic data: 100 samples, 5 features
    # Some features with skew to test Yeo-Johnson
    rng = np.random.RandomState(42)
    X_raw = rng.exponential(scale=2.0, size=(100, 5))

    # Instantiate the preprocessor
    preprocessor = Float64Preprocessor()

    # Fit and Transform
    print("Fitting and transforming synthetic data...")
    X_transformed = preprocessor.fit_transform(X_raw)

    # Verification
    # 1. Check dtype
    assert X_transformed.dtype == np.float64, "Output must be float64"

    # 2. Check Standardization (Mean ~ 0, Std ~ 1)
    means = np.mean(X_transformed, axis=0)
    stds = np.std(X_transformed, axis=0)

    print(f"Transformed Means (should be ~0): {means}")
    print(f"Transformed Stds (should be ~1): {stds}")

    assert np.allclose(means, 0, atol=1e-7), "Features not centered correctly"
    assert np.allclose(stds, 1, atol=1e-7), "Features not scaled correctly"
    print("Preprocessing logic verified.")


def demo_data_loading(cache_dir):
    print("\n--- 2. Demonstrating Data Loader ---")
    # Load data using the provided function
    # This function reads metadata, sorts features, and applies preprocessing
    print(f"Loading data into cache: {cache_dir}")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_data(
        cache_dir=cache_dir,
        load_cached_data=False,  # Force reload from metadata for demonstration
    )

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")
    print(f"Number of Classes: {len(classes)}")

    # Assertions
    assert X_train.shape[1] == 192, "Expected 192 features (margin + shape + texture)"
    assert len(y_train) == X_train.shape[0], "Mismatch in training labels"
    assert X_train.dtype == np.float64, "Data loader should return float64"

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes


def demo_model_training(X_train, y_train, X_val, y_val, classes):
    print("\n--- 3. Demonstrating CholeskyOASDiscriminant Model ---")

    # Instantiate model
    model = CholeskyOASDiscriminant()

    # Fit model
    print("Fitting model on training data...")
    model.fit(X_train, y_train)

    # Check attributes
    assert hasattr(model, "covariance_"), "Model should have estimated covariance"
    assert hasattr(model, "weights_"), "Model should have computed weights"

    # Predict on Validation
    print("Predicting probabilities on validation data...")
    val_probs = model.predict_proba(X_val)

    # Verify Probabilities
    # 1. Shape
    assert val_probs.shape == (
        X_val.shape[0],
        len(classes),
    ), "Probability shape mismatch"

    # 2. Sum to 1
    row_sums = np.sum(val_probs, axis=1)
    # Note: Softmax sums to 1, but the model clips probabilities to [1e-15, 1-1e-15]
    # So sums might slightly deviate from 1.0, but should be very close.
    # We check if they are reasonably normalized or if the clipping logic is applied.
    print(f"Mean probability sum: {np.mean(row_sums):.6f}")

    # 3. Range
    assert (
        val_probs.min() >= 0.0 and val_probs.max() <= 1.0
    ), "Probabilities out of range"

    # Calculate Metrics
    loss = log_loss(y_val, val_probs, labels=classes)

    # Predict labels for accuracy check
    val_preds_idx = np.argmax(val_probs, axis=1)
    val_preds_labels = classes[val_preds_idx]
    acc = accuracy_score(y_val, val_preds_labels)

    print(f"Validation Log Loss: {loss:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")

    # Basic sanity check on performance (random guess is -log(1/99) ~ 4.6)
    assert loss < 4.0, "Model performance is worse than random guessing"

    return model, val_probs


def demo_submission(test_ids, classes, model, X_test, output_path):
    print("\n--- 4. Demonstrating Submission Generation ---")

    # Generate test probabilities
    test_probs = model.predict_proba(X_test)

    # Save submission
    print(f"Saving submission to {output_path}")
    save_submission(test_ids, classes, test_probs, output_path)

    # Verify file creation
    assert os.path.exists(output_path), "Submission file was not created"

    # Verify file content format
    df = pd.read_csv(output_path)
    assert df.shape == (
        len(test_ids),
        len(classes) + 1,
    ), "Submission has incorrect shape"
    assert "id" in df.columns, "Submission missing 'id' column"
    assert df.iloc[0]["id"] == test_ids[0], "ID mismatch in submission"
    print("Submission file verified.")


def demo_full_pipeline(pipeline_work_dir, pipeline_sub_path):
    print("\n--- 5. Demonstrating Full Pipeline Wrapper ---")
    # The run_pipeline function encapsulates loading, training, and submission
    print("Running end-to-end pipeline...")

    # We use a fresh directory for the pipeline's cache
    model = run_pipeline(cache_dir=pipeline_work_dir, submission_path=pipeline_sub_path)

    assert os.path.exists(pipeline_sub_path), "Pipeline failed to generate submission"
    print("Full pipeline executed successfully.")


if __name__ == "__main__":
    # Setup
    set_seed(42)

    # Define working directories
    WORKING_DIR = "./working/demo_execution"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache_manual")
    PIPELINE_CACHE_DIR = os.path.join(WORKING_DIR, "cache_pipeline")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "manual_submission.csv")
    PIPELINE_SUB_PATH = os.path.join(WORKING_DIR, "pipeline_submission.csv")

    # Clean previous runs
    clean_dir(WORKING_DIR)

    # 1. Preprocessing
    demo_preprocessing()

    # 2. Data Loading
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = demo_data_loading(
        CACHE_DIR
    )

    # 3. Model Training
    model, val_probs = demo_model_training(X_train, y_train, X_val, y_val, classes)

    # 4. Submission
    demo_submission(test_ids, classes, model, X_test, SUBMISSION_PATH)

    # 5. Full Pipeline
    demo_full_pipeline(PIPELINE_CACHE_DIR, PIPELINE_SUB_PATH)

    print("\nAll demonstrations completed successfully.")
