import os
import shutil
import numpy as np
import pandas as pd
import sys
import joblib

# Import the provided library modules
import library.config as config
import library.data_loader as data_loader
import library.feature_extraction as feature_extraction
import library.training_engine as training_engine
import library.inference_engine as inference_engine
import library.model_factory as model_factory


def setup_demo_environment():
    """
    Creates a isolated demo directory and patches library modules to use it.
    This ensures we don't overwrite main artifacts and allows for a fast, clean run.
    """
    # Define demo paths
    DEMO_ROOT = "./working/demo_run"
    DEMO_CACHE = os.path.join(DEMO_ROOT, "cache")
    DEMO_MODELS = os.path.join(DEMO_ROOT, "models")
    DEMO_SUBMISSION_DIR = os.path.join(DEMO_ROOT, "submission")
    DEMO_SUBMISSION_PATH = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")

    # Clean and create directories
    if os.path.exists(DEMO_ROOT):
        shutil.rmtree(DEMO_ROOT)
    os.makedirs(DEMO_CACHE, exist_ok=True)
    os.makedirs(DEMO_MODELS, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    print(f"Initialized demo environment at {DEMO_ROOT}")

    # Monkey-patch the modules to use these new paths
    # We must patch the specific variables imported in each module

    # Patch data_loader
    data_loader.WORKING_DIR = DEMO_ROOT

    # Patch feature_extraction
    feature_extraction.CACHE_DIR = DEMO_CACHE

    # Patch training_engine
    training_engine.MODEL_DIR = DEMO_MODELS

    # Patch inference_engine
    inference_engine.MODEL_DIR = DEMO_MODELS
    inference_engine.SUBMISSION_PATH = DEMO_SUBMISSION_PATH

    return DEMO_ROOT, DEMO_SUBMISSION_PATH


def run_demo():
    # Set seeds for reproducibility
    np.random.seed(42)

    demo_root, submission_path = setup_demo_environment()

    print("\n=== STEP 1: Data Loading & Subsampling ===")
    # Load full data using the library function
    # We disable cache loading to force a fresh read from metadata
    df_train_full = data_loader.load_dataset("train", load_cached_data=False)
    df_test_full = data_loader.load_dataset("test", load_cached_data=False)

    # Subsample for speed (Logic Verification)
    # We take 50 samples for training and 10 for testing
    df_train = df_train_full.head(50).copy().reset_index(drop=True)
    df_test = df_test_full.head(10).copy().reset_index(drop=True)

    print(f"Subsampled Train Shape: {df_train.shape}")
    print(f"Subsampled Test Shape: {df_test.shape}")

    # Verify required columns exist
    assert (
        "text_combined" in df_train.columns
    ), "Preprocessing failed to create 'text_combined'"
    assert (
        "subreddit_string" in df_train.columns
    ), "Preprocessing failed to create 'subreddit_string'"

    print("\n=== STEP 2: Feature Pipeline Execution ===")
    # Instantiate pipeline (disable cache loading to ensure we compute on our subset)
    pipeline = feature_extraction.FeaturePipeline(load_cached_data=False)

    # Fit and Transform on Train
    print("Running fit_transform on training data...")
    X_lex, X_comm, X_sem, X_meta = pipeline.fit_transform(df_train)

    # Validation of Feature Shapes
    print(f"Lexical Shape: {X_lex.shape}")
    print(f"Community Shape: {X_comm.shape}")
    print(f"Semantic Shape: {X_sem.shape}")
    print(f"Metadata Shape: {X_meta.shape}")

    assert X_lex.shape[0] == 50, "Lexical features row count mismatch"
    assert (
        X_sem.shape[1] == 384
    ), "Semantic embeddings should have 384 dimensions (MiniLM)"
    assert X_meta.shape[1] == len(
        config.METADATA_FEATURES
    ), "Metadata feature count mismatch"

    print("\n=== STEP 3: Hybrid Training (Debug Mode) ===")
    # Initialize Trainer in Debug Mode (Fast execution: fewer trees, fewer iterations)
    trainer = training_engine.HybridTrainer(debug=True)

    # Train
    trainer.train(df_train, pipeline)

    # Verify Model Artifacts
    expected_models = [
        "lexical_bagger.joblib",  # Stable
        "community_bagger.joblib",  # Stable
        "semantic_booster_fold_0.joblib",  # Volatile (Fold 0)
        "semantic_booster_fold_1.joblib",  # Volatile (Fold 1)
        "meta_learner.joblib",
    ]

    print("Verifying saved models...")
    for model_file in expected_models:
        path = os.path.join(training_engine.MODEL_DIR, model_file)
        if not os.path.exists(path):
            # Note: Some models might not be in the list if they are stable/volatile differently
            # but we check a few key ones.
            print(
                f"Warning: {model_file} not found (might be expected depending on config)."
            )
        else:
            print(f"Confirmed: {model_file} exists.")

    # Check specifically for meta learner
    assert os.path.exists(
        os.path.join(training_engine.MODEL_DIR, "meta_learner.joblib")
    ), "Meta learner not saved"

    print("\n=== STEP 4: Inference ===")
    # Initialize Predictor
    # CRITICAL: We must set n_folds=2 because the debug trainer only trained 2 folds for volatile models.
    # If we use default (5), it will look for fold_2, fold_3... and fail.
    predictor = inference_engine.HybridPredictor(n_folds=2)

    # Run Prediction
    submission_df = predictor.predict(df_test, pipeline)

    print("\n=== STEP 5: Output Validation ===")
    print(submission_df.head())

    # Check Schema
    assert list(submission_df.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Incorrect submission columns"

    # Check Row Count
    assert (
        len(submission_df) == 10
    ), f"Expected 10 predictions, got {len(submission_df)}"

    # Check Probabilities
    probs = submission_df["requester_received_pizza"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"
    assert probs.dtype == float, "Probabilities should be floats"

    # Save manually to verify file writing (though predictor usually does it if we call generate_submission)
    # The predictor.predict returns a DF, we save it to the patched path.
    submission_df.to_csv(submission_path, index=False)
    assert os.path.exists(submission_path), "Submission file not created on disk"

    print(f"\nSUCCESS: Demo completed. Output saved to {submission_path}")


if __name__ == "__main__":
    run_demo()
