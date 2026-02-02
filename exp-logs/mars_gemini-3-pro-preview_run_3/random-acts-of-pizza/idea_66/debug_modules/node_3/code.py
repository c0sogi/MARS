import os
import sys
import pandas as pd
import numpy as np
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import from the provided library
from library.config import Config
from library.pipeline import run_training_pipeline, run_inference_pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_config():
    """
    Overrides the default configuration with lightweight parameters
    to ensure the demonstration runs quickly.
    """
    print("Setting up lightweight configuration for demo...")

    # Reduce Cross-Validation folds
    Config.N_FOLDS = 2

    # Reduce Ensemble Estimators for Speed
    Config.LEXICAL_BAGGER_PARAMS["n_estimators"] = 5
    Config.COMMUNITY_BAGGER_PARAMS["n_estimators"] = 5
    Config.SEMANTIC_BAGGER_PARAMS["n_estimators"] = 5

    # Reduce Boosting Iterations
    Config.SEMANTIC_BOOSTER_PARAMS["n_estimators"] = 10
    Config.SEMANTIC_BOOSTER_PARAMS["early_stopping_rounds"] = 5

    Config.SEMANTIC_GRADIENT_PARAMS["n_estimators"] = 10
    Config.SEMANTIC_GRADIENT_PARAMS["early_stopping_rounds"] = 5

    Config.TEMPORAL_BOOSTER_PARAMS["n_estimators"] = 10
    Config.TEMPORAL_BOOSTER_PARAMS["early_stopping_rounds"] = 5

    # Ensure cache directory is set to a demo specific location to avoid conflicts
    Config.CACHE_DIR = "./working/demo_run_cache"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set fixed seeds for reproducibility
    np.random.seed(Config.RANDOM_SEED)


def validate_outputs():
    """
    Validates that the pipeline produced the expected artifacts.
    """
    print("\nValidating pipeline outputs...")

    # 1. Check Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not created at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")

    # Check columns
    expected_cols = [Config.ID_COL, Config.TARGET_COL]
    if not all(col in df_sub.columns for col in expected_cols):
        raise AssertionError(
            f"Submission missing required columns. Found: {df_sub.columns}"
        )

    # Check values are probabilities
    if not ((df_sub[Config.TARGET_COL] >= 0) & (df_sub[Config.TARGET_COL] <= 1)).all():
        raise AssertionError("Predictions contain values outside [0, 1] range.")

    # 2. Check Model Artifacts
    models_dir = os.path.join(Config.CACHE_DIR, "models")
    expected_models = [
        "lexical_bagger_fold_0.joblib",
        "community_bagger_fold_0.joblib",
        "semantic_booster_fold_0.joblib",
        "meta_learner.joblib",
    ]

    for model_file in expected_models:
        path = os.path.join(models_dir, model_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected model artifact missing: {path}")

    print("Validation successful! All artifacts present and valid.")


if __name__ == "__main__":
    # 1. Configure for Speed
    setup_demo_config()

    # Define debug size (number of samples)
    DEBUG_SIZE = 50

    try:
        # 2. Run Training Pipeline
        # We set load_cached_data=False to force feature generation for this demo run
        print(
            f"\n{'='*20} Running Training Pipeline (Debug Size: {DEBUG_SIZE}) {'='*20}"
        )
        run_training_pipeline(debug_size=DEBUG_SIZE, load_cached_data=False)

        # 3. Run Inference Pipeline
        print(
            f"\n{'='*20} Running Inference Pipeline (Debug Size: {DEBUG_SIZE}) {'='*20}"
        )
        run_inference_pipeline(debug_size=DEBUG_SIZE, load_cached_data=True)

        # 4. Validate Results
        validate_outputs()

        print("\nDemo execution completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        raise e
