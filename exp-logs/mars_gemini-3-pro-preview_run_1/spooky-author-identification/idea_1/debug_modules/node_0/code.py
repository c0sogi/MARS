import os
import sys
import numpy as np
import pandas as pd
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import load_data
from library.feature_engineering import extract_features
from library.model_definition import get_logistic_regression_model
from library.training_engine import run_cross_validation, train_and_predict


def set_seed(seed=42):
    """Sets random seed for reproducibility."""
    np.random.seed(seed)


def main():
    print("=== Starting Library Usage Demonstration ===\n")
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Isolation
    # -------------------------------------------------------------------------
    print("1. Configuring environment for fast demonstration...")

    # Enable Debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Use only 100 samples per split

    # Isolate cache and submission to a demo directory to avoid conflicts
    Config.CACHE_DIR = "./working/demo_cache/"
    Config.SUBMISSION_DIR = "./working/demo_submission/"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Reduce Feature Complexity for speed
    Config.WORD_TFIDF_PARAMS["max_features"] = 50  # Very small vocab
    Config.CHAR_TFIDF_PARAMS["max_features"] = 50

    # Reduce Model Complexity for speed
    Config.MODEL_PARAMS["max_iter"] = 20

    # Clean up demo directories if they exist from previous runs
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)

    print(f"   Debug Mode: {Config.DEBUG}")
    print(f"   Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"   Cache Dir: {Config.CACHE_DIR}")
    print("   Configuration complete.\n")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("2. Demonstrating Data Loader...")
    train_df, val_df, test_df = load_data(debug=Config.DEBUG, load_cached_data=False)

    # Verification
    assert len(train_df) == Config.DEBUG_SAMPLE_SIZE, "Train set size mismatch."
    assert len(val_df) == Config.DEBUG_SAMPLE_SIZE, "Validation set size mismatch."
    assert len(test_df) == Config.DEBUG_SAMPLE_SIZE, "Test set size mismatch."

    required_cols = {"id", "text", "author"}
    assert required_cols.issubset(
        train_df.columns
    ), "Train DF missing required columns."
    assert "author" not in test_df.columns, "Test DF should not have 'author' column."

    print(f"   Successfully loaded {len(train_df)} training samples.")
    print("   Data Loading verification passed.\n")

    # -------------------------------------------------------------------------
    # 3. Feature Engineering
    # -------------------------------------------------------------------------
    print("3. Demonstrating Feature Extraction...")
    # Note: We pass load_cached_data=True, but since we cleared the dir, it will compute from scratch.
    X_train, y_train, X_val, y_val, X_test, classes = extract_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    # Verification
    # Expected features = word_max_features + char_max_features
    expected_features = (
        Config.WORD_TFIDF_PARAMS["max_features"]
        + Config.CHAR_TFIDF_PARAMS["max_features"]
    )

    assert X_train.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        expected_features,
    ), f"X_train shape mismatch. Expected ({Config.DEBUG_SAMPLE_SIZE}, {expected_features}), got {X_train.shape}"
    assert y_train.shape == (Config.DEBUG_SAMPLE_SIZE,), "y_train shape mismatch."
    assert len(classes) == 3, "Expected 3 classes (EAP, HPL, MWS)."
    assert isinstance(X_train, np.ndarray), "X_train should be a dense numpy array."

    print(f"   Generated feature matrix shape: {X_train.shape}")
    print(f"   Classes detected: {classes}")
    print("   Feature Extraction verification passed.\n")

    # -------------------------------------------------------------------------
    # 4. Model Definition
    # -------------------------------------------------------------------------
    print("4. Demonstrating Model Definition...")
    model = get_logistic_regression_model()

    # Verification
    assert (
        model.max_iter == Config.MODEL_PARAMS["max_iter"]
    ), "Model parameter override failed."
    assert (
        model.solver == Config.MODEL_PARAMS["solver"]
    ), "Model default parameter missing."

    print(f"   Model initialized: {model}")
    print("   Model Definition verification passed.\n")

    # -------------------------------------------------------------------------
    # 5. Training Engine - Cross Validation
    # -------------------------------------------------------------------------
    print("5. Demonstrating Cross Validation...")
    # Using 2 folds for speed
    cv_score = run_cross_validation(X_train, y_train, n_splits=2)

    # Verification
    assert isinstance(cv_score, float), "CV score should be a float."
    assert cv_score > 0, "Log loss should be positive."

    print(f"   CV Score (Log Loss): {cv_score:.4f}")
    print("   Cross Validation verification passed.\n")

    # -------------------------------------------------------------------------
    # 6. Training Engine - Final Prediction & Submission
    # -------------------------------------------------------------------------
    print("6. Demonstrating Training and Prediction...")

    # We need the IDs from the test dataframe for the submission file
    test_ids = test_df["id"]

    submission_df = train_and_predict(X_train, y_train, X_test, classes, test_ids)

    # Verification
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Check structure
    loaded_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(loaded_sub) == Config.DEBUG_SAMPLE_SIZE, "Submission row count mismatch."
    assert "id" in loaded_sub.columns, "Submission missing 'id' column."
    for cls in classes:
        assert cls in loaded_sub.columns, f"Submission missing class column '{cls}'."

    # Check probability validity
    probs = loaded_sub[classes].values
    # Note: LogisticRegression predict_proba sums to 1, but let's check range
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of [0, 1] range."

    print(f"   Submission saved to: {Config.SUBMISSION_PATH}")
    print(f"   Submission head:\n{submission_df.head(3)}")
    print("   Training and Prediction verification passed.\n")

    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
