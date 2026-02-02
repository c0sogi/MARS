import os
import pandas as pd
import numpy as np
import warnings

# Import provided library components
from library.config import Config
from library.utils import set_seed
from library.training_pipeline import TrainingPipeline
from library.inference_pipeline import InferencePipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Initializing Demo Script...")

    # =========================================================================
    # 1. Configuration Override
    # =========================================================================
    # We patch the Config class attributes to optimize for speed and use a
    # specific working directory for this demo.

    DEMO_DIR = "./working/demo_run"
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.MODEL_DIR = os.path.join(DEMO_DIR, "models")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce hyperparameters for rapid execution
    Config.N_FOLDS = 2
    Config.EARLY_STOPPING_ROUNDS = 2
    Config.VOCAB_SIZE_COMMUNITY = 50  # Small vocab for TF-IDF

    # Minimal estimators for all models
    Config.LEXICAL_BAGGER_PARAMS["n_estimators"] = 5
    Config.COMMUNITY_BAGGER_PARAMS["n_estimators"] = 5
    Config.SEMANTIC_BOOSTER_PARAMS["n_estimators"] = 5
    Config.SEMANTIC_GRADIENT_PARAMS["n_estimators"] = 5
    Config.SEMANTIC_BAGGER_PARAMS["n_estimators"] = 5
    Config.TEMPORAL_BOOSTER_PARAMS["n_estimators"] = 5
    Config.METADATA_ANCHOR_PARAMS["max_iter"] = 10

    # Set seed for reproducibility
    set_seed(Config.RANDOM_STATE)

    # =========================================================================
    # 2. Training Pipeline Execution
    # =========================================================================
    print("\n--- Starting Training Pipeline (Debug Mode) ---")
    trainer = TrainingPipeline()

    # Run with a small subset (debug_size=50) to verify logic quickly.
    # load_cached_data=False ensures we compute features from scratch for this demo run.
    trainer.run(load_cached_data=False, debug_size=50)

    # =========================================================================
    # 3. Validation of Training Artifacts
    # =========================================================================
    print("\n--- Validating Training Artifacts ---")

    # Check OOF Predictions
    oof_path = os.path.join(Config.WORKING_DIR, "oof_predictions.csv")
    if not os.path.exists(oof_path):
        raise FileNotFoundError(f"OOF predictions file missing at {oof_path}")

    oof_df = pd.read_csv(oof_path)
    print(f"OOF Shape: {oof_df.shape}")

    # Verify we have predictions for the debug subset (50 samples)
    if len(oof_df) != 50:
        raise AssertionError(f"Expected 50 OOF predictions, found {len(oof_df)}")

    # Verify all Level-1 models are present as columns
    expected_models = [
        "lexical_bagger",
        "community_bagger",
        "semantic_booster",
        "semantic_gradient",
        "semantic_bagger",
        "metadata_anchor",
        "temporal_booster",
    ]
    for model in expected_models:
        if model not in oof_df.columns:
            raise AssertionError(f"Missing model column in OOF: {model}")

    # Check Model Files
    # We expect:
    # 1. Meta Learner
    # 2. Fold models for Volatile learners (e.g., semantic_booster_fold_0)
    # 3. Full models for Stable learners (e.g., lexical_bagger)

    meta_path = os.path.join(Config.MODEL_DIR, "meta_learner.joblib")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Meta-Learner model file missing.")

    # Check a specific stable model
    stable_model_path = os.path.join(Config.MODEL_DIR, "lexical_bagger.joblib")
    if not os.path.exists(stable_model_path):
        raise FileNotFoundError("Stable model (Lexical Bagger) missing.")

    # Check a specific volatile fold model
    volatile_fold_path = os.path.join(
        Config.MODEL_DIR, "semantic_booster_fold_0.joblib"
    )
    if not os.path.exists(volatile_fold_path):
        raise FileNotFoundError(
            "Volatile fold model (Semantic Booster Fold 0) missing."
        )

    print("Training artifacts validated successfully.")

    # =========================================================================
    # 4. Inference Pipeline Execution
    # =========================================================================
    print("\n--- Starting Inference Pipeline ---")
    inferencer = InferencePipeline()

    # We use load_cached_data=True to reuse the features generated during training.
    # Since we use the same debug_size and seed, the data split is identical.
    submission_df = inferencer.run(load_cached_data=True, debug_size=50)

    # =========================================================================
    # 5. Validation of Submission
    # =========================================================================
    print("\n--- Validating Submission ---")

    # Check object type
    if not isinstance(submission_df, pd.DataFrame):
        raise TypeError("Inference pipeline did not return a DataFrame.")

    # Check shape (should match debug_size)
    if len(submission_df) != 50:
        raise AssertionError(f"Expected 50 submission rows, found {len(submission_df)}")

    # Check required columns
    if Config.ID_COL not in submission_df.columns:
        raise AssertionError(f"Submission missing ID column: {Config.ID_COL}")
    if Config.TARGET_COL not in submission_df.columns:
        raise AssertionError(f"Submission missing Target column: {Config.TARGET_COL}")

    # Check probability validity
    probs = submission_df[Config.TARGET_COL]
    if probs.min() < 0.0 or probs.max() > 1.0:
        raise AssertionError("Predicted probabilities are out of [0, 1] range.")

    # Check file persistence
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print(f"Submission generated at: {Config.SUBMISSION_PATH}")
    print("\nAll demonstrations and validations passed.")


if __name__ == "__main__":
    main()
