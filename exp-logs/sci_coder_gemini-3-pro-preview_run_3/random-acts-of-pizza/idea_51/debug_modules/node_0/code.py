import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import feature_engineering
from library import models
from library import engine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def override_config_for_demo():
    """
    Modifies the global configuration to ensure the demo runs quickly.
    Reduces CV folds, estimators, and iterations.
    """
    print("[Demo] Overriding configuration for speed...")

    # Reduce Cross-Validation Folds
    config.N_FOLDS = 2

    # Reduce computational complexity for all models
    for model_name, params in config.MODEL_PARAMS.items():
        # Tree-based models
        if "n_estimators" in params:
            params["n_estimators"] = 10  # drastically reduce from 200/2000

        # Gradient Boosting specific
        if "early_stopping_rounds" in params:
            params["early_stopping_rounds"] = 5

        # Linear models
        if "max_iter" in params:
            params["max_iter"] = 50

    # Reduce Vectorizer features for speed
    config.LEXICAL_VECTORIZER_PARAMS["max_features"] = 1000
    config.COMMUNITY_VECTORIZER_PARAMS["max_features"] = 100


def validate_feature_data(data):
    """
    Validates the integrity of the generated feature dictionary.
    """
    print("[Demo] Validating generated features...")

    required_keys = [
        "X_train_lexical",
        "X_test_lexical",
        "X_train_behavioral",
        "X_test_behavioral",
        "X_train_semantic",
        "X_test_semantic",
        "X_train_contextual",
        "X_test_contextual",
        "y_train",
        "train_ids",
        "test_ids",
    ]

    for key in required_keys:
        if key not in data:
            raise AssertionError(f"Missing key in feature data: {key}")

    n_train = len(data["y_train"])
    n_test = len(data["test_ids"])

    # Check dimensions
    assert data["X_train_lexical"].shape[0] == n_train, "Lexical Train rows mismatch"
    assert data["X_test_lexical"].shape[0] == n_test, "Lexical Test rows mismatch"
    assert (
        data["X_train_contextual"].shape[0] == n_train
    ), "Contextual Train rows mismatch"

    # Check types (Sparse vs Dense)
    assert sp.issparse(data["X_train_lexical"]), "Lexical features should be sparse"
    assert isinstance(
        data["X_train_semantic"], np.ndarray
    ), "Semantic features should be dense numpy array"

    print(f"[Demo] Validation Passed. Train samples: {n_train}, Test samples: {n_test}")


def main():
    # 1. Setup
    utils.set_seed(42)
    override_config_for_demo()

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 2. Feature Engineering Pipeline
    # We set load_cached_data=False to demonstrate the generation logic
    print("\n" + "=" * 40)
    print("STEP 1: Feature Engineering")
    print("=" * 40)

    pipeline = feature_engineering.FeaturePipeline(load_cached_data=False)
    data = pipeline.run()

    # Validate the output
    validate_feature_data(data)

    # 3. Model Training & Ensemble
    print("\n" + "=" * 40)
    print("STEP 2: Ensemble Training (Hept-View)")
    print("=" * 40)

    eng = engine.EnsembleEngine(data)

    # Run Cross-Validation and Base Learner Training
    eng.run_cv_and_training()

    # Verify OOF Matrix
    assert not eng.oof_matrix.empty, "OOF Matrix should not be empty after training"
    assert eng.oof_matrix.shape[0] == len(
        data["y_train"]
    ), "OOF Matrix row count mismatch"
    print(f"[Demo] OOF Matrix shape: {eng.oof_matrix.shape}")

    # Train Level 2 Meta-Learner
    eng.train_meta_learner()

    # 4. Submission Generation
    print("\n" + "=" * 40)
    print("STEP 3: Inference & Submission")
    print("=" * 40)

    eng.generate_submission()

    # 5. Final Validation
    if os.path.exists(config.SUBMISSION_PATH):
        df_sub = pd.read_csv(config.SUBMISSION_PATH)
        print(f"[Demo] Submission file created at: {config.SUBMISSION_PATH}")
        print(f"[Demo] Submission shape: {df_sub.shape}")

        # Check columns
        expected_cols = ["request_id", "requester_received_pizza"]
        if list(df_sub.columns) == expected_cols:
            print("[Demo] Submission columns verified.")
        else:
            raise AssertionError(
                f"Invalid columns. Expected {expected_cols}, got {list(df_sub.columns)}"
            )

        # Check ID consistency
        if set(df_sub["request_id"]) == set(data["test_ids"]):
            print("[Demo] Submission IDs match test set.")
        else:
            raise AssertionError("Submission IDs do not match test set IDs.")

    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n[Demo] Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
