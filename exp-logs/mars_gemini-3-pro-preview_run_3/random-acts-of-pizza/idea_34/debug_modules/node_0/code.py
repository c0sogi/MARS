import os
import sys
import shutil
import pandas as pd
import numpy as np
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
# We assume the script is running from the root directory where 'library' is a package
from library import config
from library import utils
from library import data_processing
from library import feature_engineering
from library import ensemble_pipeline


def configure_demo_settings():
    """
    Overrides default configuration parameters to ensure the demo runs quickly.
    Reduces model complexity, feature counts, and cross-validation folds.
    """
    print("Configuring demo settings for rapid execution...")

    # 1. Paths
    config.CACHE_DIR = "./working/demo_run_cache/"
    config.SUBMISSION_DIR = "./working/demo_submission/"
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # 2. Feature Engineering Limits
    config.TFIDF_PARAMS["max_features"] = 100  # Reduce vocabulary size
    config.COMMUNITY_MAX_FEATURES = 50  # Reduce community features

    # 3. Model Hyperparameters (Low estimators for speed)
    base_rf_params = {
        "n_estimators": 5,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "n_jobs": 1,  # Avoid overhead for small data
        "random_state": config.RANDOM_STATE,
        "verbose": 0,
    }

    config.INTERACTION_BAGGER_PARAMS = base_rf_params.copy()
    config.LEXICAL_BAGGER_PARAMS = base_rf_params.copy()
    config.COMMUNITY_BAGGER_PARAMS = base_rf_params.copy()
    config.SEMANTIC_BAGGER_PARAMS = base_rf_params.copy()

    config.SEMANTIC_BOOSTER_PARAMS = {
        "n_estimators": 10,
        "learning_rate": 0.1,
        "max_depth": 2,
        "n_jobs": 1,
        "random_state": config.RANDOM_STATE,
        "verbosity": 0,
        "early_stopping_rounds": 5,
    }

    config.METADATA_ANCHOR_PARAMS = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": config.RANDOM_STATE,
        "max_iter": 100,
    }

    # 4. Pipeline Settings
    config.N_FOLDS = 2  # Minimal folds for CV


def validate_features(feats_dict, name):
    """
    Validates the structure and content of the generated feature dictionaries.
    """
    required_keys = ["holistic", "lexical", "community", "semantic", "metadata", "y"]
    for key in required_keys:
        if key not in feats_dict:
            raise AssertionError(f"Missing key '{key}' in {name} features.")

        arr = feats_dict[key]
        if not isinstance(arr, np.ndarray):
            raise AssertionError(f"Feature '{key}' in {name} is not a numpy array.")

        if len(arr) == 0:
            raise AssertionError(f"Feature '{key}' in {name} is empty.")

    print(
        f"Validation passed for {name} features. Shape of holistic: {feats_dict['holistic'].shape}"
    )


def main():
    # 1. Setup
    utils.set_seed(42)
    configure_demo_settings()

    utils.print_header("Step 1: Data Loading & Processing")
    # Load raw data, clean text, process metadata
    # We force reprocessing to demonstrate the logic (load_cached_data=False)
    # In a real run, we might set this to True.
    train_df, val_df, test_df = data_processing.load_and_process_data(
        load_cached_data=False
    )

    # Basic assertions on dataframes
    assert not train_df.empty, "Train dataframe is empty"
    assert "holistic_text" in train_df.columns, "Holistic text column missing"
    assert config.TARGET_COL in train_df.columns, "Target column missing in train"
    print(
        f"Data loaded. Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
    )

    utils.print_header("Step 2: Feature Engineering")
    # Initialize engineer
    fe = feature_engineering.FeatureEngineer()

    # Generate features (Sparse TF-IDF, Dense Embeddings, Metadata)
    # We force recomputation (load_cached_data=False)
    train_feats, val_feats, test_feats = fe.generate_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Validate feature structures
    validate_features(train_feats, "Train")
    validate_features(val_feats, "Val")
    validate_features(test_feats, "Test")

    utils.print_header("Step 3: Ensemble Training (CV)")
    # Initialize the stacking ensemble
    ensemble = ensemble_pipeline.StackingEnsemble(train_feats, val_feats, test_feats)

    # Run Cross-Validation to train Level 2 Meta-Learner
    # This trains base models on (K-1) folds and predicts on the hold-out fold
    ensemble.train_cv(n_folds=config.N_FOLDS)

    # Check if OOF predictions were generated
    assert ensemble.oof_preds is not None, "OOF predictions not generated."
    assert ensemble.oof_preds.shape[0] == len(
        train_df
    ), "OOF predictions shape mismatch."

    utils.print_header("Step 4: Retraining Final Base Models")
    # Retrain base models on the full training set (Train + Val usually, or just Train depending on logic)
    # The pipeline logic combines Train + Val for RF/Linear, uses Val for early stopping in XGB
    ensemble.train_final_models()

    # Check if models are stored
    expected_models = [
        "interaction_bagger",
        "lexical_bagger",
        "community_bagger",
        "semantic_booster",
        "semantic_bagger",
        "metadata_anchor",
    ]
    for name in expected_models:
        if name not in ensemble.final_base_models:
            raise AssertionError(f"Final model '{name}' was not trained.")

    utils.print_header("Step 5: Prediction & Submission")
    # Generate predictions on Test set
    ensemble.predict()

    # Validate Submission
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    submission = pd.read_csv(config.SUBMISSION_PATH)

    # Check dimensions
    assert len(submission) == len(
        test_df
    ), f"Submission row count mismatch. Expected {len(test_df)}, got {len(submission)}"
    assert list(submission.columns) == [
        config.ID_COL,
        config.TARGET_COL,
    ], "Submission columns mismatch"

    # Check probability range
    probs = submission[config.TARGET_COL]
    if probs.min() < 0 or probs.max() > 1:
        raise AssertionError("Predictions contain values outside [0, 1] range.")

    print("\nSUCCESS: Pipeline executed successfully and submission generated.")
    print(submission.head())


if __name__ == "__main__":
    main()
