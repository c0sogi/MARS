import os
import sys
import pandas as pd
import numpy as np
import shutil
from sklearn.pipeline import Pipeline

# Import from the provided library
from library.config import Config
from library.utils import set_seed, load_object
from library.data_loader import DataLoader
from library.features import get_feature_pipeline
from library.model import get_model, tune_hyperparameters
from library.trainer import run_cv_training
from library.inference import generate_submission


def main():
    print("Starting demonstration of Random Acts of Pizza pipeline...")

    # ==========================================
    # 1. Configuration Patching for Speed
    # ==========================================
    print("\n[1] Patching Configuration for Rapid Execution...")
    # Modify Config class attributes directly to ensure fast execution for the demo
    Config.DEBUG = True
    Config.MAX_SAMPLES = 50  # Use only 50 samples
    Config.N_FOLDS = 2  # Only 2 folds for CV
    Config.N_BAGGING_ESTIMATORS = 2  # Minimal ensemble size
    Config.C_GRID = [0.1, 1.0]  # Reduced search space
    Config.CLASS_WEIGHT_GRID = [None]  # Reduced search space

    # Ensure directories exist
    Config.setup()
    set_seed(Config.SEED)
    print("Configuration patched successfully.")

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n[2] Verifying Data Loader...")
    loader = DataLoader()

    # Test loading training data
    df_train = loader.load_merged_data(split="train", load_cached_data=False)
    print(f"Loaded training data shape: {df_train.shape}")

    # Assertions
    assert (
        len(df_train) == Config.MAX_SAMPLES
    ), f"Expected {Config.MAX_SAMPLES} samples in debug mode, got {len(df_train)}"
    assert "requester_received_pizza" in df_train.columns, "Target column missing"
    assert (
        "text_combined" in df_train.columns
    ), "Text preprocessing failed (text_combined missing)"
    assert (
        "requester_subreddits_at_request" in df_train.columns
    ), "Subreddit list column missing"

    # Test loading test data
    df_test = loader.load_merged_data(split="test", load_cached_data=False)
    print(f"Loaded test data shape: {df_test.shape}")
    assert len(df_test) == Config.MAX_SAMPLES, "Test data subsampling failed"

    print("Data Loader verification passed.")

    # ==========================================
    # 3. Feature Engineering Verification
    # ==========================================
    print("\n[3] Verifying Feature Pipeline...")

    # Separate features and target
    X_train = df_train.drop(columns=["requester_received_pizza"])
    y_train = df_train["requester_received_pizza"]

    # Initialize pipeline
    feature_pipe = get_feature_pipeline()

    # Fit and Transform
    print("Fitting feature pipeline...")
    X_transformed = feature_pipe.fit_transform(X_train, y_train)

    print(f"Transformed feature shape: {X_transformed.shape}")

    # Assertions
    assert (
        X_transformed.shape[0] == Config.MAX_SAMPLES
    ), "Row count mismatch after transformation"
    # We expect:
    # - 384 dims from SBERT
    # - Config.TOP_K_LEXICAL (50) from TF-IDF
    # - Config.TOP_K_COMMUNITY (20) from Community TF-IDF
    # - Polynomial features from metadata (variable depending on degree and input cols)
    # Just checking it's not empty and has substantial width
    assert (
        X_transformed.shape[1] > 400
    ), f"Feature dimension too low: {X_transformed.shape[1]}"

    print("Feature Pipeline verification passed.")

    # ==========================================
    # 4. Model Architecture & Tuning Verification
    # ==========================================
    print("\n[4] Verifying Model and Hyperparameter Tuning...")

    # Test Model Instantiation
    model = get_model(C=1.0, class_weight=None)
    assert (
        model.n_estimators == Config.N_BAGGING_ESTIMATORS
    ), "Bagging estimator count mismatch"

    # Test Hyperparameter Tuning (Grid Search)
    # This uses the pipeline internally
    print("Running mini hyperparameter tuning...")
    best_params = tune_hyperparameters(X_train, y_train)

    print(f"Best parameters found: {best_params}")
    assert "C" in best_params, "Tuning did not return 'C'"
    assert "class_weight" in best_params, "Tuning did not return 'class_weight'"

    print("Model and Tuning verification passed.")

    # ==========================================
    # 5. Full Training Loop Verification
    # ==========================================
    print("\n[5] Executing Cross-Validation Training Loop...")

    # This runs the full workflow defined in trainer.py
    fold_scores = run_cv_training()

    # Assertions
    assert (
        len(fold_scores) == Config.N_FOLDS
    ), f"Expected {Config.N_FOLDS} scores, got {len(fold_scores)}"
    assert all(isinstance(s, float) for s in fold_scores), "Scores are not floats"

    # Verify artifacts were saved
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"fold_{fold}_pipeline.joblib")
        assert os.path.exists(model_path), f"Model artifact for fold {fold} missing"

    print("Training loop execution passed.")

    # ==========================================
    # 6. Inference and Submission Verification
    # ==========================================
    print("\n[6] Generating Submission...")

    # Run inference
    generate_submission(
        working_dir=Config.WORKING_DIR, submission_path=Config.SUBMISSION_PATH
    )

    # Assertions
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    df_submission = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_submission.shape}")

    # Check format
    assert list(df_submission.columns) == [
        "request_id",
        "requester_received_pizza",
    ], f"Incorrect submission columns: {df_submission.columns}"
    assert (
        len(df_submission) == Config.MAX_SAMPLES
    ), f"Submission row count mismatch. Expected {Config.MAX_SAMPLES}, got {len(df_submission)}"

    # Check probability values
    preds = df_submission["requester_received_pizza"]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("Inference and Submission verification passed.")

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
