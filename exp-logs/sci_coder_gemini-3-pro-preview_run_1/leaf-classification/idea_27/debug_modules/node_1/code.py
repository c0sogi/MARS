import os
import numpy as np
import pandas as pd
import shutil
from library import config
from library import utils
from library import data_loader
from library import preprocessor
from library import model


# -----------------------------------------------------------------------------
# Setup and Configuration
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clean_working_directory():
    """Cleans the specific idea directory to ensure a fresh run."""
    if os.path.exists(config.IDEA_DIR):
        shutil.rmtree(config.IDEA_DIR)
    os.makedirs(config.IDEA_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Demonstration Functions
# -----------------------------------------------------------------------------


def demo_config():
    print("\n=== 1. Validating Configuration ===")
    print(f"Input Directory: {config.INPUT_DIR}")
    print(f"Metadata Directory: {config.METADATA_DIR}")
    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Idea Directory: {config.IDEA_DIR}")

    # Verify feature columns
    expected_features = 192
    assert (
        len(config.FEATURE_COLS) == expected_features
    ), f"Expected {expected_features} feature columns, got {len(config.FEATURE_COLS)}"
    print(f"Feature Columns Verified: {len(config.FEATURE_COLS)} features defined.")
    print("Configuration loaded successfully.")


def demo_data_loader():
    print("\n=== 2. Testing Data Loader ===")

    # Force reload from CSVs to test parsing logic
    # We pass load_cached_data=False to ignore any existing cache
    data = data_loader.load_datasets(load_cached_data=False)

    # Validate keys
    expected_keys = [
        "X_train",
        "y_train",
        "X_val",
        "y_val",
        "X_test",
        "ids_test",
        "classes",
    ]
    for key in expected_keys:
        assert key in data, f"Missing key in loaded data: {key}"

    # Validate shapes
    n_train = data["X_train"].shape[0]
    n_val = data["X_val"].shape[0]
    n_features = data["X_train"].shape[1]

    print(f"Training Samples: {n_train}")
    print(f"Validation Samples: {n_val}")
    print(f"Features: {n_features}")
    print(f"Classes: {len(data['classes'])}")

    assert n_features == 192, "Incorrect number of features in X_train"
    assert data["y_train"].shape[0] == n_train, "Mismatch between X_train and y_train"
    assert data["X_train"].dtype == np.float64, "X_train must be float64"

    print("Data Loader test passed.")
    return data


def demo_preprocessor():
    print("\n=== 3. Testing Preprocessor (Float64Transformer) ===")

    # 1. Unit Test on Dummy Data
    # Create random data with different scales to test standardization
    dummy_X = np.random.randn(100, 5) * 10 + 5

    transformer = preprocessor.Float64Transformer()
    transformer.fit(dummy_X)
    transformed_X = transformer.transform(dummy_X)

    # Check precision
    assert transformed_X.dtype == np.float64, "Transformer output must be float64"

    # Check standardization properties (Mean ~ 0, Std ~ 1)
    means = np.mean(transformed_X, axis=0)
    stds = np.std(transformed_X, axis=0)

    print(f"Dummy Data - Mean of transformed features: {means}")
    print(f"Dummy Data - Std of transformed features: {stds}")

    assert np.allclose(means, 0, atol=1e-7), "Transformed means should be close to 0"
    assert np.allclose(stds, 1, atol=1e-7), "Transformed stds should be close to 1"

    # 2. Integration Test with Real Data Pipeline
    print("Running process_and_cache_data()...")
    # This runs the full loading -> transformation -> caching pipeline
    processed_data = preprocessor.process_and_cache_data(load_cached_data=False)

    assert os.path.exists(
        os.path.join(config.IDEA_DIR, "X_train_transformed.npy")
    ), "Transformed training data file not created."

    print("Preprocessor test passed.")
    return processed_data


def demo_model_logic(processed_data):
    print("\n=== 4. Testing DualExpertLDA Model Logic ===")

    X_train = processed_data["X_train"]
    y_train = processed_data["y_train"]
    X_val = processed_data["X_val"]

    # Instantiate model
    clf = model.DualExpertLDA(empirical_jitter=1e-5)

    # Fit
    print("Fitting model...")
    clf.fit(X_train, y_train)

    # Predict
    print("Predicting probabilities...")
    probas = clf.predict_proba(X_val)
    preds = clf.predict(X_val)

    # Validations
    assert probas.shape == (
        X_val.shape[0],
        len(processed_data["classes"]),
    ), "Probability matrix shape mismatch"

    # Check row sums equal 1
    row_sums = np.sum(probas, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), "Probabilities do not sum to 1"

    print(f"Prediction shape: {probas.shape}")
    print("DualExpertLDA logic test passed.")
    return probas


def demo_utils(processed_data, val_probas):
    print("\n=== 5. Testing Utilities ===")

    y_val = processed_data["y_val"]
    ids_test = processed_data["ids_test"]
    classes = processed_data["classes"]

    # 1. Calculate Log Loss
    loss = utils.calculate_log_loss(y_val, val_probas)
    print(f"Calculated Validation Log Loss: {loss:.5f}")
    assert loss > 0, "Log loss should be positive"

    # 2. Save Submission
    # Create dummy probabilities for test set for demonstration
    n_test = len(ids_test)
    n_classes = len(classes)
    dummy_test_probas = np.full((n_test, n_classes), 1.0 / n_classes)

    output_dir = os.path.join(config.WORKING_DIR, "demo_submission")
    utils.save_submission(ids_test, dummy_test_probas, classes, output_dir=output_dir)

    submission_path = os.path.join(output_dir, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify file content format
    df_sub = pd.read_csv(submission_path)
    assert df_sub.shape == (n_test, n_classes + 1), "Submission CSV has incorrect shape"
    assert "id" in df_sub.columns, "Submission CSV missing 'id' column"

    print("Utilities test passed.")


def demo_full_pipeline():
    print("\n=== 6. Running Full Training Pipeline ===")

    # The run_training_pipeline function in model.py encapsulates the entire workflow:
    # Load -> Process -> Train -> Eval -> Submit
    val_loss = model.run_training_pipeline(load_cached_data=True)

    print(f"Pipeline completed. Final Validation Loss: {val_loss:.5f}")

    # Verify final submission exists in the default location
    expected_sub_path = "./submission/submission.csv"
    if os.path.exists(expected_sub_path):
        print(f"Final submission found at {expected_sub_path}")
    else:
        raise FileNotFoundError(f"Final submission not found at {expected_sub_path}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # Clean up previous runs to ensure tests run on fresh state where appropriate
    clean_working_directory()

    try:
        # 1. Config
        demo_config()

        # 2. Data Loader
        # We load raw data first
        raw_data_dict = demo_data_loader()

        # 3. Preprocessor
        # This transforms the raw data and caches it
        processed_data_dict = demo_preprocessor()

        # 4. Model Logic
        # We use the processed data to test the model class specifically
        val_probabilities = demo_model_logic(processed_data_dict)

        # 5. Utils
        # We verify metric calculation and submission file generation
        demo_utils(processed_data_dict, val_probabilities)

        # 6. Full Pipeline
        # We run the provided pipeline function to verify end-to-end integration
        demo_full_pipeline()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
