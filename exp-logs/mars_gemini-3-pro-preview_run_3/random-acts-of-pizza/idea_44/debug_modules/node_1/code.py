import os
import sys
import numpy as np
import pandas as pd
import warnings
from unittest.mock import MagicMock

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library import config
from library import utils
from library import data_loader
from library import feature_engineering
from library import ensemble_trainer


def demonstrate_pipeline():
    logger = utils.get_logger("DemoScript")
    logger.info("Starting Pipeline Demonstration...")

    # =========================================================================
    # 1. OPTIMIZATION & CONFIGURATION OVERRIDES
    # =========================================================================
    logger.info("Overriding configuration parameters for speed...")

    # Reduce Cross-Validation folds
    config.N_FOLDS = 2

    # Reduce Feature Dimensionality
    config.TFIDF_PARAMS["max_features"] = 50
    config.SUBREDDIT_TFIDF_PARAMS["max_features"] = 20

    # Reduce Model Complexity (Estimators, Depth, Early Stopping)
    # Lexical Bagger
    config.RF_LEXICAL_PARAMS["n_estimators"] = 5
    config.RF_LEXICAL_PARAMS["n_jobs"] = 1

    # Community Bagger
    config.RF_COMMUNITY_PARAMS["n_estimators"] = 5
    config.RF_COMMUNITY_PARAMS["n_jobs"] = 1

    # Semantic Booster (XGBoost)
    config.XGB_SEMANTIC_PARAMS["n_estimators"] = 5
    config.XGB_SEMANTIC_PARAMS["early_stopping_rounds"] = 1
    config.XGB_SEMANTIC_PARAMS["n_jobs"] = 1

    # Semantic Bagger
    config.RF_SEMANTIC_PARAMS["n_estimators"] = 5
    config.RF_SEMANTIC_PARAMS["n_jobs"] = 1

    # Temporal Booster (LightGBM)
    config.LGBM_TEMPORAL_PARAMS["n_estimators"] = 5
    config.LGBM_TEMPORAL_PARAMS["n_jobs"] = 1

    # Set Seed
    utils.set_seed(config.SEED)

    # =========================================================================
    # 2. MOCKING HEAVY COMPUTE (SentenceTransformer)
    # =========================================================================
    # We mock the SentenceTransformer to avoid downloading models and running
    # heavy inference during this quick demo. We return random noise as embeddings.
    logger.info("Mocking SentenceTransformer for fast execution...")

    class MockSentenceTransformer:
        def __init__(self, model_name_or_path, device=None):
            self.model_name = model_name_or_path

        def encode(self, sentences, show_progress_bar=False, convert_to_numpy=True):
            # Return random embeddings: (n_samples, 384) - 384 is standard for MiniLM
            n_samples = len(sentences)
            return np.random.rand(n_samples, 384).astype(np.float32)

    # Apply the mock
    feature_engineering.SentenceTransformer = MockSentenceTransformer

    # =========================================================================
    # 3. DATA LOADING & FEATURE ENGINEERING
    # =========================================================================
    logger.info("Initializing DataLoader (forcing feature recalculation)...")

    # Force recalculation to test feature engineering logic
    loader = data_loader.DataLoader(load_cached_data=False)

    # Execute loading
    data = loader.load_data()

    # --- Validation: Data Structure ---
    logger.info("Validating loaded data structure...")

    required_keys = [
        "y_train",
        "y_val",
        "X_train_meta",
        "X_val_meta",
        "X_test_meta",
        "X_train_lexical",
        "X_train_behavioral",
        "X_train_semantic",
        "test_ids",
    ]

    for key in required_keys:
        if key not in data:
            raise AssertionError(f"Missing key in data dictionary: {key}")

    # Check shapes
    n_train = len(data["y_train"])
    n_val = len(data["y_val"])
    n_test = len(data["test_ids"])

    assert data["X_train_meta"].shape[0] == n_train, "X_train_meta row count mismatch"
    assert data["X_val_meta"].shape[0] == n_val, "X_val_meta row count mismatch"
    assert data["X_test_meta"].shape[0] == n_test, "X_test_meta row count mismatch"

    # Check sparse matrix handling
    assert (
        data["X_train_lexical"].shape[0] == n_train
    ), "Lexical features shape mismatch"

    logger.info(
        f"Data Loaded Successfully. Train: {n_train}, Val: {n_val}, Test: {n_test}"
    )

    # =========================================================================
    # 4. MODEL TRAINING & STACKING
    # =========================================================================
    logger.info("Initializing StackingTrainer...")
    trainer = ensemble_trainer.StackingTrainer()

    # Run the full training pipeline
    # This includes: OOF Generation -> Meta-Learner Training -> Final Retraining -> Submission
    logger.info("Running training pipeline (this may take a few seconds)...")
    trainer.run(data)

    # --- Validation: Internal State ---
    logger.info("Validating trainer state...")

    # Check if all base learners were retrained
    expected_models = [
        "lexical_bagger",
        "community_bagger",
        "semantic_booster",
        "semantic_bagger",
        "metadata_anchor",
        "temporal_booster",
    ]

    for name in expected_models:
        if name not in trainer.final_models:
            raise AssertionError(
                f"Model {name} was not retrained/stored in final_models."
            )

    logger.info("All base learners successfully trained.")

    # =========================================================================
    # 5. SUBMISSION VERIFICATION
    # =========================================================================
    logger.info("Verifying submission file...")

    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(config.SUBMISSION_PATH)

    # Check columns
    assert config.ID_COL in submission_df.columns, f"Missing ID column {config.ID_COL}"
    assert (
        config.TARGET_COL in submission_df.columns
    ), f"Missing Target column {config.TARGET_COL}"

    # Check length
    assert (
        len(submission_df) == n_test
    ), f"Submission length {len(submission_df)} != Test set size {n_test}"

    # Check value range (probabilities)
    probs = submission_df[config.TARGET_COL]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    logger.info(f"Submission verified. Shape: {submission_df.shape}")
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    demonstrate_pipeline()
