import os
import sys
import warnings
import pandas as pd
import numpy as np

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import feature_engineering
from library import model_definitions
from library import training_pipeline


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # -------------------------------------------------------------------------
    print(">>> Setting up environment and overriding configuration for demo...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    utils.set_seed(42)

    # Override config parameters for speed
    config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples
    config.N_FOLDS = 2  # Use 2-fold CV instead of 5
    config.EMBEDDING_BATCH_SIZE = 16

    # Reduce dimensionality for NMF
    config.NMF_N_COMPONENTS = 5
    config.TOP_K_SUBREDDITS = 100

    # Reduce Model Complexity (fewer trees, fewer iterations)
    config.RF_LEXICAL_PARAMS["n_estimators"] = 5
    config.RF_COMMUNITY_PARAMS["n_estimators"] = 5
    config.RF_SEMANTIC_PARAMS["n_estimators"] = 5
    config.XGB_SEMANTIC_PARAMS["n_estimators"] = 5
    config.LR_ANCHOR_PARAMS["max_iter"] = 50

    # Ensure working directory for demo exists
    demo_working_dir = "./working/demo_run"
    config.WORKING_DIR = demo_working_dir
    config.CACHE_DIR = os.path.join(demo_working_dir, "cache")
    config.MODEL_DIR = os.path.join(demo_working_dir, "models")
    config.PREDICTIONS_DIR = os.path.join(demo_working_dir, "predictions")
    config.SUBMISSION_DIR = os.path.join(demo_working_dir, "submission")
    config.SUBMISSION_FILE_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    print("Configuration overrides applied.")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Demonstrating Data Loading...")

    # Load Train/Val (limited by DEBUG_SAMPLE_SIZE)
    train_df, val_df = data_loader.load_dataset(
        "train", debug_size=config.DEBUG_SAMPLE_SIZE
    )

    # Load Test
    test_df = data_loader.load_dataset("test", debug_size=config.DEBUG_SAMPLE_SIZE)

    # Validation
    assert (
        len(train_df) == config.DEBUG_SAMPLE_SIZE
    ), f"Train size mismatch: {len(train_df)}"
    assert len(val_df) == config.DEBUG_SAMPLE_SIZE, f"Val size mismatch: {len(val_df)}"
    assert (
        len(test_df) == config.DEBUG_SAMPLE_SIZE
    ), f"Test size mismatch: {len(test_df)}"

    # Check preprocessing
    assert (
        "text_combined" in train_df.columns
    ), "Preprocessing failed: 'text_combined' missing."
    assert (
        "requester_received_pizza" in train_df.columns
    ), "Target column missing in train."

    print(
        f"Data Loaded successfully. Train shape: {train_df.shape}, Test shape: {test_df.shape}"
    )

    # -------------------------------------------------------------------------
    # 3. Feature Engineering Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Demonstrating Feature Engineering...")

    # A. Static Feature Extraction (Embeddings + Metadata)
    print("   Running StaticFeatureExtractor...")
    static_extractor = feature_engineering.StaticFeatureExtractor()

    # Extract features for training set
    # Using a unique cache prefix for this demo step to avoid conflict with pipeline
    static_train = static_extractor.extract(train_df, "demo_train_static")

    assert "embeddings" in static_train
    assert "metadata" in static_train
    assert static_train["embeddings"].shape == (
        config.DEBUG_SAMPLE_SIZE,
        384,
    ), f"Embedding shape mismatch: {static_train['embeddings'].shape}"
    assert (
        not static_train["metadata"].isnull().values.any()
    ), "Metadata contains NaNs before imputation (unexpected for selected cols)."

    # B. Dynamic Feature Extraction (TF-IDF, NMF, Scaling)
    print("   Running DynamicFeatureExtractor...")
    dynamic_extractor = feature_engineering.DynamicFeatureExtractor()

    # Fit on training data
    dynamic_extractor.fit(train_df, static_train["metadata"])

    # Transform training data
    dynamic_out = dynamic_extractor.transform(train_df, static_train["metadata"])

    # Validation of output keys
    expected_keys = [
        "X_lexical",
        "X_behavioral_sparse",
        "X_community_latent",
        "X_metadata_scaled",
    ]
    for key in expected_keys:
        assert key in dynamic_out, f"Dynamic feature output missing key: {key}"

    # Validation of shapes
    assert dynamic_out["X_lexical"].shape[0] == config.DEBUG_SAMPLE_SIZE
    assert dynamic_out["X_community_latent"].shape[1] == config.NMF_N_COMPONENTS

    print("Feature Engineering logic verified.")

    # -------------------------------------------------------------------------
    # 4. Model Definitions Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Demonstrating Model Definitions...")

    models = model_definitions.get_level1_models()

    # Verify all 5 expected branches exist
    expected_models = [
        "lexical_rf",
        "community_rf",
        "semantic_xgb",
        "semantic_rf",
        "metadata_lr",
    ]
    for m in expected_models:
        assert m in models, f"Model definition missing: {m}"

    # Verify Meta Learner
    meta_learner = model_definitions.get_meta_learner()
    assert hasattr(meta_learner, "fit"), "Meta learner is not a valid estimator."

    print(f"Models instantiated successfully. Level 1 count: {len(models)}")

    # -------------------------------------------------------------------------
    # 5. Full Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n>>> Executing Full Training Pipeline (Integration Test)...")

    # This function orchestrates data loading, feature extraction, stacking CV, retraining, and submission.
    # It uses the overridden config values (small sample size, low iterations) so it should run quickly.
    training_pipeline.run_training_pipeline(debug_size=config.DEBUG_SAMPLE_SIZE)

    # -------------------------------------------------------------------------
    # 6. Submission Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Submission Artifacts...")

    if not os.path.exists(config.SUBMISSION_FILE_PATH):
        raise FileNotFoundError(
            f"Submission file not created at {config.SUBMISSION_FILE_PATH}"
        )

    submission_df = pd.read_csv(config.SUBMISSION_FILE_PATH)

    # Check dimensions
    assert (
        len(submission_df) == config.DEBUG_SAMPLE_SIZE
    ), f"Submission length {len(submission_df)} does not match debug size {config.DEBUG_SAMPLE_SIZE}"

    # Check columns
    assert "request_id" in submission_df.columns
    assert "requester_received_pizza" in submission_df.columns

    # Check values (probabilities)
    probs = submission_df["requester_received_pizza"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of [0, 1] range."

    print(f"Submission verified at: {config.SUBMISSION_FILE_PATH}")
    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    main()
