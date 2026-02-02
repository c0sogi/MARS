import os
import sys
import numpy as np
import pandas as pd
import logging
import warnings
import shutil

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import from the provided library
from library.config import Config
from library.utils import set_seed, setup_logging
from library.data_loader import load_datasets
from library.feature_engineering import get_features
from library.training_engine import HybridTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def configure_demo_settings():
    """
    Overrides default Config parameters to ensure the demo runs quickly.
    """
    print("Configuring demo settings for speed...")

    # Reduce Cross-Validation Folds
    Config.N_FOLDS = 2

    # Reduce Vectorizer Vocabulary Sizes
    Config.TEXT_VECTORIZER_PARAMS["max_features"] = 100
    Config.COMMUNITY_VECTORIZER_PARAMS["max_features"] = 50

    # Reduce Ensemble Estimators and Iterations
    # Branch 1
    Config.LEXICAL_BAGGER_PARAMS["n_estimators"] = 5
    Config.LEXICAL_ANCHOR_PARAMS["max_iter"] = 10

    # Branch 2
    Config.COMMUNITY_BAGGER_PARAMS["n_estimators"] = 5
    Config.COMMUNITY_ANCHOR_PARAMS["max_iter"] = 10

    # Branch 3
    Config.SEMANTIC_BOOSTER_PARAMS["n_estimators"] = 5
    Config.SEMANTIC_GRADIENT_PARAMS["n_estimators"] = 5
    Config.SEMANTIC_BAGGER_PARAMS["n_estimators"] = 5

    # Branch 4
    Config.METADATA_ANCHOR_PARAMS["max_iter"] = 10
    Config.TEMPORAL_BOOSTER_PARAMS["n_estimators"] = 5

    # Meta Learner
    Config.META_LEARNER_PARAMS["max_iter"] = 10

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


def main():
    # 1. Setup
    setup_logging(log_name="demo_run.log")
    set_seed(42)
    configure_demo_settings()

    print("\n=== Step 1: Loading Data ===")
    # Load raw data using the library's data loader
    # We disable caching for the demo to ensure we run the logic fresh
    train_df, val_df, test_df = load_datasets(load_cache=False)

    # OPTIMIZATION: Subset data for speed
    SUBSET_SIZE = 100
    print(f"Subsetting data to {SUBSET_SIZE} samples for demonstration...")
    train_df = train_df.head(SUBSET_SIZE).copy()
    val_df = val_df.head(SUBSET_SIZE).copy()
    test_df = test_df.head(SUBSET_SIZE).copy()

    # Verify Data Loading
    assert not train_df.empty, "Training dataframe is empty"
    assert Config.TARGET_COL in train_df.columns, "Target column missing in Train"
    assert (
        Config.TARGET_COL not in test_df.columns
    ), "Target column should not be in Test"
    print("Data loaded and subsetted successfully.")

    print("\n=== Step 2: Feature Engineering ===")
    # Extract Features for Train
    # This will fit the pipeline and return the feature dictionary
    print("Processing Training Data...")
    train_features, pipeline = get_features(
        train_df, split_name="train", load_cache=False
    )

    # Extract Features for Validation (using fitted pipeline)
    print("Processing Validation Data...")
    val_features = get_features(
        val_df, split_name="val", pipeline=pipeline, load_cache=False
    )

    # Extract Features for Test (using fitted pipeline)
    print("Processing Test Data...")
    test_features = get_features(
        test_df, split_name="test", pipeline=pipeline, load_cache=False
    )

    # Verify Feature Shapes
    print("Verifying feature consistency...")
    expected_keys = ["X_lexical", "X_behavioral", "X_metadata", "X_semantic"]
    for key in expected_keys:
        assert key in train_features, f"Missing {key} in train features"
        assert train_features[key].shape[0] == len(train_df), f"Shape mismatch in {key}"
        assert test_features[key].shape[0] == len(test_df), f"Shape mismatch in {key}"

    print(f"Lexical Feature Shape: {train_features['X_lexical'].shape}")
    print(f"Semantic Feature Shape: {train_features['X_semantic'].shape}")

    print("\n=== Step 3: Training Stacking Ensemble ===")
    # Prepare Targets
    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values

    # Instantiate Trainer
    trainer = HybridTrainer()

    # Train Level 1 (Base Learners)
    # This handles both Stable (Full Retrain) and Volatile (CV-Bagging) models
    print("Training Level 1 Base Learners...")
    trainer.train_stacking_layer(train_features, y_train, val_features, y_val)

    # Verify Level 1 Artifacts
    model_dir = os.path.join(Config.WORKING_DIR, "models")
    assert os.path.exists(model_dir), "Model directory not created"
    # Check for a stable model
    assert os.path.exists(
        os.path.join(model_dir, "lexical_bagger.joblib")
    ), "Stable model artifact missing"

    # Train Level 2 (Meta Learner)
    print("Training Level 2 Meta Learner...")
    trainer.train_meta_learner()
    assert os.path.exists(
        os.path.join(model_dir, "meta_learner.joblib")
    ), "Meta learner artifact missing"

    print("\n=== Step 4: Inference ===")
    # Generate Predictions on Test Set
    print("Generating predictions...")
    submission_df = trainer.generate_predictions(test_features)

    # Verify Submission
    print("Verifying submission...")
    assert isinstance(submission_df, pd.DataFrame), "Submission is not a DataFrame"
    assert len(submission_df) == len(test_df), "Submission row count mismatch"
    assert Config.ID_COL in submission_df.columns, "Request ID column missing"
    assert Config.TARGET_COL in submission_df.columns, "Prediction column missing"

    # Check values are probabilities
    preds = submission_df[Config.TARGET_COL]
    assert (
        preds.min() >= 0 and preds.max() <= 1
    ), "Predictions out of probability range [0, 1]"

    # Check file existence
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found on disk"

    print("\n=== Demo Completed Successfully ===")
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print(f"Top 5 Predictions:\n{submission_df.head().to_string()}")


if __name__ == "__main__":
    main()
