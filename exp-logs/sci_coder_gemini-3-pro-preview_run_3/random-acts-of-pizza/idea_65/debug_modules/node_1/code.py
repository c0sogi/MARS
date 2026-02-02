import os
import shutil
import numpy as np
import pandas as pd
import library.config as config
from library.utils import set_seed, setup_logger
from library.data_loader import load_and_process_data
from library.feature_processor import get_processed_features
from library.trainer import Trainer
from library.predictor import Predictor


def configure_demo_environment():
    """
    Overrides default configuration to optimize for speed and separate demo artifacts.
    """
    print("Configuring demo environment...")

    # Redirect directories to a demo-specific folder
    demo_dir = "./working/demo_run"
    config.WORKING_DIR = demo_dir
    config.CACHE_DIR = os.path.join(demo_dir, "cache")
    config.MODEL_DIR = os.path.join(demo_dir, "models")
    config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Reduce computational load for demonstration
    config.N_FOLDS = 2  # Minimum for CV
    config.DEBUG = True

    # Reduce Estimators for Speed
    # Branch 1
    config.LEXICAL_BAGGER_PARAMS["n_estimators"] = 5
    config.LEXICAL_ANCHOR_PARAMS["max_iter"] = 100

    # Branch 2
    config.COMMUNITY_BAGGER_PARAMS["n_estimators"] = 5
    config.COMMUNITY_ANCHOR_PARAMS["max_iter"] = 100

    # Branch 3
    config.SEMANTIC_BOOSTER_PARAMS["n_estimators"] = 5
    config.SEMANTIC_GRADIENT_PARAMS["n_estimators"] = 5
    config.SEMANTIC_BAGGER_PARAMS["n_estimators"] = 5

    # Branch 4
    config.TEMPORAL_BOOSTER_PARAMS["n_estimators"] = 5
    config.METADATA_ANCHOR_PARAMS["max_iter"] = 100

    # Meta Learner
    config.META_LEARNER_PARAMS["max_iter"] = 100

    # Vectorizer limits (speed up fit/transform)
    config.LEXICAL_VECTORIZER_PARAMS["max_features"] = 1000
    config.COMMUNITY_VECTORIZER_PARAMS["max_features"] = 100

    # Early stopping
    config.EARLY_STOPPING_ROUNDS = 5


def validate_feature_dict(features, name="Features"):
    """Validates the structure and content of the feature dictionary."""
    required_keys = ["X_lexical", "X_community", "X_semantic", "X_metadata"]
    for key in required_keys:
        if key not in features:
            raise AssertionError(f"{name} missing key: {key}")

        # Check for NaNs in dense arrays
        if not isinstance(features[key], (pd.DataFrame, pd.Series)):
            if isinstance(features[key], np.ndarray):
                if np.isnan(features[key]).any():
                    raise AssertionError(f"{name} {key} contains NaNs")

    print(f"[{name}] Validation Passed. Keys: {list(features.keys())}")


def main():
    # 1. Setup
    set_seed(42)
    logger = setup_logger("demo_script")
    configure_demo_environment()

    logger.info("=== Step 1: Data Loading ===")
    # Load data (forcing re-process to demonstrate logic, though cache logic handles it)
    # We pass load_cached_data=False to ensure we see the processing logic in action
    train_df, test_df = load_and_process_data(load_cached_data=False)

    # Assertions
    assert not train_df.empty, "Training dataframe is empty"
    assert not test_df.empty, "Test dataframe is empty"
    assert (
        config.TARGET_COL in train_df.columns
    ), f"Target column {config.TARGET_COL} missing"
    # Check leakage removal
    leakage_cols = [c for c in train_df.columns if c.endswith("_at_retrieval")]
    assert len(leakage_cols) == 0, f"Leakage columns detected: {leakage_cols}"

    logger.info(f"Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")

    logger.info("=== Step 2: Feature Processing ===")
    # Generate features
    train_features, test_features = get_processed_features(
        train_df, test_df, load_cached_data=False
    )

    # Validate Features
    validate_feature_dict(train_features, "Train Features")
    validate_feature_dict(test_features, "Test Features")
    assert "y" in train_features, "Target 'y' missing from train features"
    assert (
        len(train_features["y"]) == train_features["X_metadata"].shape[0]
    ), "Mismatch in X and y length"

    logger.info("=== Step 3: Model Training (Ensemble) ===")
    trainer = Trainer()

    # Extract test IDs for submission generation later
    test_ids = test_df[config.ID_COL].values

    # Run Training
    trainer.train_ensemble(train_features, test_features, test_ids)

    # Validate Artifacts
    expected_models = [
        "meta_learner.joblib",
        "lexical_bagger.joblib",  # Stable model
        "community_bagger.joblib",  # Stable model
        "semantic_booster_fold_0.joblib",  # Volatile model fold 0
        "semantic_booster_fold_1.joblib",  # Volatile model fold 1
    ]

    logger.info("Validating model artifacts...")
    for model_file in expected_models:
        path = os.path.join(config.MODEL_DIR, model_file)
        if not os.path.exists(path):
            raise AssertionError(f"Expected model artifact missing: {path}")
    print("Model artifacts verified.")

    logger.info("=== Step 4: Inference ===")
    predictor = Predictor()

    # Run Prediction
    submission_df = predictor.predict_ensemble(test_features, test_ids)

    logger.info("=== Step 5: Final Validation ===")
    # 1. Check File Existence
    if not os.path.exists(config.SUBMISSION_PATH):
        raise AssertionError(f"Submission file not found at {config.SUBMISSION_PATH}")

    # 2. Check Shape
    expected_rows = len(test_df)
    if len(submission_df) != expected_rows:
        raise AssertionError(
            f"Submission has {len(submission_df)} rows, expected {expected_rows}"
        )

    # 3. Check Columns
    expected_cols = [config.ID_COL, config.TARGET_COL]
    if list(submission_df.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch. Got {submission_df.columns}, expected {expected_cols}"
        )

    # 4. Check Probabilities
    probs = submission_df[config.TARGET_COL]
    if probs.min() < 0 or probs.max() > 1:
        raise AssertionError("Predictions contain values outside [0, 1] range")

    # 5. Check IDs match
    if not np.array_equal(submission_df[config.ID_COL].values, test_ids):
        raise AssertionError("Submission IDs do not match Test IDs")

    print("\n" + "=" * 30)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print(f"Submission generated at: {config.SUBMISSION_PATH}")
    print(f"Sample predictions:\n{submission_df.head()}")
    print("=" * 30)


if __name__ == "__main__":
    main()
