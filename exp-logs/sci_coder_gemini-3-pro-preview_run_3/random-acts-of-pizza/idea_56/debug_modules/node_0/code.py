import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.hybrid_ensemble import HybridStackingEnsemble
from library.utils import set_seed


def configure_for_speed():
    """
    Monkeypatches the Config class to optimize for speed during this demonstration.
    Reduces dataset size, fold count, and model complexity.
    """
    print("Configuring parameters for rapid demonstration...")

    # 1. Enable Debug Mode (Uses only 500 rows)
    Config.DEBUG = True

    # 2. Reduce Cross-Validation Folds
    Config.N_FOLDS = 2

    # 3. Reduce Complexity of Base Learners
    # We iterate through all attributes starting with HP_ (Hyperparameters)
    # and reduce n_estimators or iterations.
    for attr_name in dir(Config):
        if attr_name.startswith("HP_") and isinstance(getattr(Config, attr_name), dict):
            hp_dict = getattr(Config, attr_name)

            # Reduce Tree Estimators
            if "n_estimators" in hp_dict:
                hp_dict["n_estimators"] = 10

            # Reduce Boosting Iterations/Early Stopping
            if "early_stopping_rounds" in hp_dict:
                hp_dict["early_stopping_rounds"] = 5

            # Reduce Complexity
            if "max_depth" in hp_dict and hp_dict["max_depth"] is not None:
                hp_dict["max_depth"] = 2

            # Reduce LightGBM specific
            if "num_leaves" in hp_dict:
                hp_dict["num_leaves"] = 5

    # 4. Set a specific cache directory for this demo run to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = "./working/demo_run/submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


def validate_ensemble(ensemble):
    """
    Validates the internal state of the ensemble after training.
    """
    print("\nValidating Ensemble Logic...")

    # 1. Check OOF Predictions
    assert ensemble.oof_preds is not None, "OOF predictions should be generated."
    assert not ensemble.oof_preds.empty, "OOF DataFrame should not be empty."
    # In DEBUG mode with 500 rows, OOF should match that count
    assert (
        len(ensemble.oof_preds) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} OOF predictions, got {len(ensemble.oof_preds)}"

    print("  [PASS] OOF Predictions generated with correct shape.")

    # 2. Check Trained Models Structure
    # We check one Stable model (e.g., lexical_bagger) and one Volatile model (e.g., semantic_booster)

    # Lexical Bagger is Stable -> Should have 'full' model, 'folds' list might be empty or populated depending on implementation detail
    # (In the provided code, stable models are cloned and fit in CV loop but not persisted in 'folds', then retrained in 'full')
    lex_bagger = ensemble.trained_models.get("lexical_bagger")
    assert lex_bagger is not None
    assert (
        lex_bagger["full"] is not None
    ), "Stable model 'lexical_bagger' should have a retrained 'full' estimator."

    # Semantic Booster is Volatile -> Should have 'folds' populated, 'full' should be None
    sem_booster = ensemble.trained_models.get("semantic_booster")
    assert sem_booster is not None
    assert (
        len(sem_booster["folds"]) == Config.N_FOLDS
    ), f"Volatile model 'semantic_booster' should have {Config.N_FOLDS} fold models."
    assert (
        sem_booster["full"] is None
    ), "Volatile model 'semantic_booster' should NOT have a 'full' estimator."

    print("  [PASS] Hybrid Training Protocol (Stable vs Volatile) verified.")


def validate_submission():
    """
    Validates the generated submission file.
    """
    print("\nValidating Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["request_id", "requester_received_pizza"]
    assert (
        list(df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df.columns)}"

    # Check rows (Debug mode applies to test set loading as well in utils.load_data)
    assert (
        len(df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows in submission, got {len(df)}"

    # Check probabilities
    probs = df["requester_received_pizza"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of [0, 1] range."

    print("  [PASS] Submission file format and content verified.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)
    configure_for_speed()

    print("=" * 40)
    print("Deca-View Ensemble Demonstration")
    print("=" * 40)

    # 2. Instantiate Ensemble
    print("\nInitializing Ensemble...")
    ensemble = HybridStackingEnsemble()

    # 3. Train (Fit)
    # We set load_cached_data=False to ensure the code actually runs the feature generation logic
    print("\n--- Phase 1: Training & Feature Engineering ---")
    ensemble.fit(load_cached_data=False)

    # 4. Validate Training State
    validate_ensemble(ensemble)

    # 5. Inference (Predict)
    print("\n--- Phase 2: Inference ---")
    preds = ensemble.predict(load_cached_data=False)

    # 6. Validate Output
    validate_submission()

    print("\n" + "=" * 40)
    print("Demonstration Completed Successfully")
    print("=" * 40)
