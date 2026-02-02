import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, Timer
from library.data_factory import DataFactory
from library.feature_engine import FeatureGenerator
from library.training_engine import CrossValidationTrainer
from library.inference_engine import HybridPredictor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def configure_for_demo():
    """
    Modifies the Config class in-place to ensure the demonstration runs quickly.
    Reduces estimator counts and folds.
    """
    print("Configuring system for fast demonstration mode...")

    # 1. Reduce Cross-Validation Folds
    Config.N_FOLDS = 2

    # 2. Reduce Tree-based Model Complexity (Level 1 Base Learners)

    # Lexical Bagger (Random Forest)
    Config.LEXICAL_RF_PARAMS.update({"n_estimators": 5, "n_jobs": -1})

    # Community Bagger (Random Forest)
    Config.COMMUNITY_RF_PARAMS.update({"n_estimators": 5, "n_jobs": -1})

    # Semantic Booster (XGBoost)
    Config.SEMANTIC_XGB_PARAMS.update({"n_estimators": 10, "n_jobs": -1})

    # Semantic Gradient (LightGBM)
    Config.SEMANTIC_LGBM_PARAMS.update({"n_estimators": 10, "n_jobs": -1})

    # Semantic Bagger (Random Forest)
    Config.SEMANTIC_RF_PARAMS.update({"n_estimators": 5, "n_jobs": -1})

    # Temporal Booster (LightGBM)
    Config.METADATA_BOOSTER_PARAMS.update({"n_estimators": 10, "n_jobs": -1})

    # Metadata Anchor (Logistic Regression) - limit iterations
    Config.METADATA_ANCHOR_PARAMS.update({"max_iter": 50})

    # 3. Reduce Training Constraints
    Config.EARLY_STOPPING_ROUNDS = 5

    # 4. Set a demo-specific working directory to avoid stale cache issues
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    Config.setup()


def verify_data_and_features():
    """
    Demonstrates and verifies DataFactory and FeatureGenerator usage.
    """
    print("\n=== Verifying DataFactory and FeatureGenerator ===")

    # 1. Load Data
    # We force reload to ensure we test the processing logic
    train_df, test_df = DataFactory.load_union_dataset(load_cached_data=False)

    print(f"Train/Val Union Shape: {train_df.shape}")
    print(f"Test Shape: {test_df.shape}")

    # Assertions
    assert not train_df.empty, "Training dataframe is empty"
    assert not test_df.empty, "Testing dataframe is empty"
    assert Config.TARGET_COL in train_df.columns, "Target column missing in train"
    assert "text_combined" in train_df.columns, "Feature engineering (text) failed"

    # 2. Generate Features
    fg = FeatureGenerator(train_df, test_df)

    # Test Lexical Features (Sparse)
    X_train_lex, X_test_lex = fg.get_lexical_features(load_cached_data=False)
    print(f"Lexical Feature Shape (Train): {X_train_lex.shape}")
    assert X_train_lex.shape[0] == train_df.shape[0], "Lexical train rows mismatch"
    assert X_test_lex.shape[0] == test_df.shape[0], "Lexical test rows mismatch"

    # Test Metadata Features (Dense)
    X_train_meta, X_test_meta = fg.get_metadata_features(load_cached_data=False)
    print(f"Metadata Feature Shape (Train): {X_train_meta.shape}")
    assert X_train_meta.shape[1] == len(
        Config.METADATA_COLS
    ), "Metadata column count mismatch"

    print("Data and Feature verification passed.")


def run_training_pipeline():
    """
    Demonstrates the CrossValidationTrainer.
    """
    print("\n=== Running Training Pipeline ===")

    trainer = CrossValidationTrainer()
    trainer.run()

    # Verify Model Artifacts
    print("Verifying model artifacts...")
    expected_models = [
        "lexical_bagger_full.joblib",  # Stable model
        "semantic_booster_fold_0.joblib",  # Volatile model (Fold 0)
        "meta_learner.joblib",  # Level 2 model
    ]

    for model_file in expected_models:
        path = os.path.join(Config.WORKING_DIR, model_file)
        if os.path.exists(path):
            print(f"[OK] Found {model_file}")
        else:
            # Note: Depending on the specific logic in TrainingEngine,
            # some files might be named differently or created only if specific branches run.
            # However, based on the provided code, these should exist.
            print(f"[WARNING] Could not find {model_file}")


def run_inference_pipeline():
    """
    Demonstrates the HybridPredictor.
    """
    print("\n=== Running Inference Pipeline ===")

    predictor = HybridPredictor()
    predictor.predict()

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated at {Config.SUBMISSION_PATH}")
        print(f"Submission Shape: {sub.shape}")

        # Check format
        assert list(sub.columns) == [
            "request_id",
            "requester_received_pizza",
        ], "Invalid submission columns"
        assert len(sub) == 1162, f"Expected 1162 predictions, got {len(sub)}"

        # Check probabilities
        probs = sub["requester_received_pizza"]
        assert (
            probs.min() >= 0.0 and probs.max() <= 1.0
        ), "Probabilities out of bounds [0, 1]"

        print("Inference verification passed.")
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # Set global seed for reproducibility
    set_seed(42)

    # 1. Configure for Speed
    configure_for_demo()

    with Timer("Total Demonstration"):
        # 2. Verify Data Loading & Feature Engineering
        verify_data_and_features()

        # 3. Run Training (Level 1 & Level 2)
        run_training_pipeline()

        # 4. Run Inference
        run_inference_pipeline()

    print("\nAll demonstration steps completed successfully.")
