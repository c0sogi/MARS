import os
import sys
import shutil
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, Timer, print_header
from library.ensemble import HexStackingEngine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def configure_demo_settings():
    """
    Modifies the global Config class to optimize for speed during this demonstration.
    Reduces the number of estimators, folds, and feature dimensions.
    """
    print_header("Configuring Demo Settings")

    # 1. Reduce Cross-Validation Folds
    Config.N_FOLDS = 2
    print(f"Set N_FOLDS to {Config.N_FOLDS}")

    # 2. Reduce Feature Dimensions for Speed
    Config.TFIDF_PARAMS["max_features"] = 100  # Reduced from 3000
    Config.PCA_COMPONENTS = 10  # Reduced from 50
    print("Reduced TF-IDF max_features and PCA components.")

    # 3. Reduce Model Complexity (Estimators)
    # Random Forests
    rf_params = {"n_estimators": 10, "n_jobs": -1, "verbose": 0}
    Config.MODEL_LEXICAL_RF.update(rf_params)
    Config.MODEL_COMMUNITY_RF.update(rf_params)
    Config.MODEL_SEMANTIC_RF.update(rf_params)

    # XGBoost
    Config.MODEL_SEMANTIC_XGB["n_estimators"] = 10

    # KNN
    Config.MODEL_MANIFOLD_KNN["n_neighbors"] = 5

    print("Reduced model estimators and neighbors.")

    # 4. Redirect Outputs to a Demo Directory
    # This ensures we don't overwrite any existing 'production' runs in ./working
    Config.CACHE_DIR = "./working/demo_run/cache/"
    Config.SUBMISSION_DIR = "./working/demo_submission/"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up demo directories if they exist to prove fresh execution
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    print(
        f"Output directories set to:\n  Cache: {Config.CACHE_DIR}\n  Submission: {Config.SUBMISSION_PATH}"
    )


def validate_submission():
    """
    Validates the generated submission file against requirements.
    """
    print_header("Validating Submission")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Loaded submission with shape: {df.shape}")

    # Check 1: Row count (Test set has 1162 rows)
    expected_rows = 1162
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"

    # Check 2: Columns
    expected_cols = [Config.ID_COL, Config.TARGET_COL]
    assert (
        list(df.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(df.columns)}"

    # Check 3: ID uniqueness
    assert df[Config.ID_COL].nunique() == expected_rows, "Request IDs are not unique"

    # Check 4: Probability bounds
    probs = df[Config.TARGET_COL]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Predictions are not valid probabilities [0, 1]"

    # Check 5: Check for NaNs
    assert not df.isnull().any().any(), "Submission contains NaN values"

    print("All validation checks passed successfully.")


def run_demo():
    """
    Main execution flow for the demonstration.
    """
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Apply speed optimizations
    configure_demo_settings()

    # Initialize the Engine
    # The engine handles feature generation, OOF CV, meta-training, and final prediction.
    engine = HexStackingEngine()

    # Execute the pipeline
    # We set load_cached_data=False to force feature generation for this demo,
    # proving the pipeline works from scratch.
    with Timer("Full Demo Pipeline"):
        # 1. Generate Out-of-Fold Predictions (Level 1)
        # This trains base models on N_FOLDS splits
        oof_matrix, y_train = engine.generate_oof(load_cached_data=False)

        # Verify OOF shape
        expected_train_size = 2302  # Based on metadata analysis
        assert (
            oof_matrix.shape[0] == expected_train_size
        ), f"OOF matrix rows mismatch. Expected {expected_train_size}, got {oof_matrix.shape[0]}"
        assert oof_matrix.shape[1] == len(
            engine.model_names
        ), "OOF matrix columns mismatch."

        # 2. Train Meta-Learner (Level 2)
        engine.train_meta_learner(oof_matrix, y_train)

        # Verify Meta-Learner is fitted
        assert hasattr(engine.meta_learner, "coef_"), "Meta-learner was not fitted."

        # 3. Retrain Base Models on Full Train+Val
        engine.retrain_base_models(load_cached_data=True)  # Can use cache now

        # Verify base models are stored
        assert len(engine.base_models) == len(
            engine.model_names
        ), "Not all base models were retrained."

        # 4. Generate Final Predictions
        submission = engine.predict(load_cached_data=False)

    # Validate the final output
    validate_submission()


if __name__ == "__main__":
    try:
        run_demo()
        print("\nDemo completed successfully.")
    except AssertionError as e:
        print(f"\n[FAILURE] Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILURE] An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
