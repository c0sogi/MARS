import os
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import (
    SEED,
    WORKING_DIR,
    SUBMISSION_DIR,
    FEATURE_COLUMNS,
    FLOAT_TYPE,
)
from library.data_pipeline import load_dataset, RobustPreprocessor
from library.model import CholeskyOASDiscriminant, run_cholesky_oas_workflow
from library.utils import calculate_log_loss, save_submission


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def demo_robust_preprocessor():
    print("\n--- Demo: RobustPreprocessor ---")
    # Create synthetic data (float32 to test casting)
    data = np.random.rand(10, 5).astype(np.float32)
    df = pd.DataFrame(data, columns=[f"feat_{i}" for i in range(5)])

    preprocessor = RobustPreprocessor()

    # Test fit_transform
    transformed_df = preprocessor.fit_transform(df)

    # Validation 1: Check Output Type
    assert transformed_df.values.dtype == np.dtype(
        FLOAT_TYPE
    ), f"Preprocessor failed to cast to {FLOAT_TYPE}"

    # Validation 2: Check Shape
    assert (
        transformed_df.shape == df.shape
    ), "Preprocessor changed the shape of the data."

    # Validation 3: Check Standardization (Mean approx 0, Std approx 1)
    # Note: PowerTransformer + StandardScaler results in mean~0, std~1
    means = transformed_df.mean()
    stds = transformed_df.std()

    assert np.allclose(means, 0, atol=1e-6), "Transformed features are not centered."
    assert np.allclose(stds, 1, atol=1e-6), "Transformed features are not scaled."

    print("RobustPreprocessor logic verified successfully.")


def demo_data_loading():
    print("\n--- Demo: Data Loading (Debug Mode) ---")
    # Load a small subset of data
    sample_size = 50
    X_train, y_train, X_val, y_val, X_test, ids_test, classes = load_dataset(
        load_cached_data=False,  # Force reload for demo
        debug=True,
        debug_sample_size=sample_size,
    )

    # Validation 1: Check Sample Sizes
    # Note: Stratified split in metadata generation might result in slightly fewer samples
    # if the total debug size is split, but here load_dataset slices *after* reading metadata.
    # The metadata contains pre-split files. load_dataset slices each file to debug_sample_size.
    assert (
        len(X_train) == sample_size
    ), f"Expected {sample_size} train samples, got {len(X_train)}"
    assert (
        len(X_val) == sample_size
    ), f"Expected {sample_size} val samples, got {len(X_val)}"
    assert (
        len(X_test) == sample_size
    ), f"Expected {sample_size} test samples, got {len(X_test)}"

    # Validation 2: Check Feature Count
    expected_features = len(FEATURE_COLUMNS)  # 192
    assert (
        X_train.shape[1] == expected_features
    ), f"Expected {expected_features} features, got {X_train.shape[1]}"

    # Validation 3: Check Label Consistency
    assert len(y_train) == len(X_train), "Mismatch between X_train and y_train length."
    assert len(classes) > 0, "No classes detected."

    print(f"Data loaded successfully. Classes found: {len(classes)}")
    return X_train, y_train, X_val, y_val, classes


def demo_model_training_and_inference(X_train, y_train, X_val, y_val, classes):
    print("\n--- Demo: CholeskyOASDiscriminant Model ---")

    model = CholeskyOASDiscriminant()

    # 1. Fit
    print("Fitting model...")
    model.fit(X_train, y_train)

    # Validation: Check attributes
    assert hasattr(model, "coef_"), "Model failed to generate coefficients."
    assert hasattr(model, "covariance_"), "Model failed to estimate covariance."
    assert model.coef_.shape == (
        len(classes),
        X_train.shape[1],
    ), f"Coefficient shape mismatch. Expected {(len(classes), X_train.shape[1])}, got {model.coef_.shape}"

    # 2. Predict Proba
    print("Predicting probabilities...")
    probs = model.predict_proba(X_val)

    # Validation: Check Probabilities
    assert probs.shape == (
        len(X_val),
        len(classes),
    ), "Probability output shape mismatch."
    assert np.all(probs >= 0) and np.all(
        probs <= 1
    ), "Probabilities out of range [0, 1]."

    # Check row sums (should be approx 1.0 due to softmax)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities do not sum to 1."

    # 3. Calculate Loss
    print("Calculating Log Loss...")
    loss = calculate_log_loss(y_val, probs, class_names=classes)
    print(f"Calculated Log Loss: {loss:.4f}")

    assert isinstance(loss, float), "Loss is not a float."
    assert loss >= 0, "Loss cannot be negative."

    print("Model training and inference verified.")


def demo_full_workflow():
    print("\n--- Demo: Full Workflow Execution ---")
    # This runs the pipeline defined in library.model
    # It handles loading, training, predicting, and saving submission
    val_loss = run_cholesky_oas_workflow(debug=True, debug_sample_size=20)

    # Check if submission file was created
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify submission content
    df_sub = pd.read_csv(submission_path)
    assert "id" in df_sub.columns, "Submission missing 'id' column."
    assert (
        len(df_sub) == 20
    ), "Submission has incorrect number of rows (should match debug size)."

    print(f"Workflow finished. Validation Loss: {val_loss:.4f}")
    print(f"Submission saved to: {submission_path}")


if __name__ == "__main__":
    set_seed(SEED)

    # Clean up working directory for fresh start (optional, but good for demo)
    if os.path.exists(WORKING_DIR):
        # We don't delete the whole dir to avoid permission issues, just specific cache files if needed
        # But load_dataset(debug=True) ignores cache, so we are safe.
        pass

    try:
        # 1. Test Preprocessing Logic
        demo_robust_preprocessor()

        # 2. Test Data Loading
        X_train, y_train, X_val, y_val, classes = demo_data_loading()

        # 3. Test Model Logic
        demo_model_training_and_inference(X_train, y_train, X_val, y_val, classes)

        # 4. Test Full Workflow
        demo_full_workflow()

        print("\nAll demonstrations completed successfully!")

    except AssertionError as e:
        print(f"\nDEMO FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        exit(1)
