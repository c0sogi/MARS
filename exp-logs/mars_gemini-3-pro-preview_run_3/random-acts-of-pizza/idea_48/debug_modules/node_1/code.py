import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import warnings
import shutil

# Import provided library modules
import library.config as config
from library.feature_engineering import FeatureProcessor
from library.model_definitions import ModelFactory
from library.ensemble_pipeline import HexViewEnsemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_demo_config():
    """
    Monkey-patch the config module to use lightweight hyperparameters
    and a separate working directory for the demo.
    """
    print("Setting up demo configuration...")

    # 1. Override Working Directory
    config.WORKING_DIR = "./working/demo_execution"
    config.SUBMISSION_DIR = os.path.join(config.WORKING_DIR, "submission")
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Re-create directories since we changed the path
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # 2. Override Cross-Validation Folds
    config.N_FOLDS = 2

    # 3. Override Model Hyperparameters for Speed
    # Reduce estimators and relax constraints for quick fitting

    # Lexical Bagger
    config.RF_PARAMS_LEXICAL.update(
        {"n_estimators": 5, "n_jobs": 1}  # Reduce thread contention in demo
    )

    # Community Bagger
    config.RF_PARAMS_COMMUNITY.update({"n_estimators": 5, "n_jobs": 1})

    # Semantic Booster (XGBoost)
    config.XGB_PARAMS_SEMANTIC.update(
        {
            "n_estimators": 5,
            "n_jobs": 1,
            "early_stopping_rounds": None,  # Disable for simple fit checks or keep low
        }
    )

    # Semantic Bagger
    config.RF_PARAMS_SEMANTIC.update({"n_estimators": 5, "n_jobs": 1})

    # Temporal Booster (LightGBM)
    config.LGBM_PARAMS_TEMPORAL.update({"n_estimators": 5, "n_jobs": 1, "verbose": -1})

    print(f"Demo Working Directory: {config.WORKING_DIR}")
    print(f"Demo Folds: {config.N_FOLDS}")


def demo_feature_engineering():
    """
    Demonstrates and validates the FeatureProcessor class.
    """
    print("\n" + "=" * 40)
    print("DEMO: Feature Engineering")
    print("=" * 40)

    processor = FeatureProcessor()

    # Run processor (force re-compute by setting load_cached_data=False initially if needed,
    # but here we rely on the clean dir created in setup)
    print("Running FeatureProcessor...")
    data = processor.run(load_cached_data=False)

    # --- Validation ---
    print("Validating FeatureProcessor output...")

    # 1. Check Top-Level Keys
    expected_splits = ["train", "val", "test"]
    for split in expected_splits:
        assert split in data, f"Missing split '{split}' in data dictionary."

    # 2. Check Feature Views in Train
    train_data = data["train"]
    expected_views = ["X_lexical", "X_behavioral", "X_semantic", "X_meta", "y"]
    for view in expected_views:
        assert view in train_data, f"Missing view '{view}' in train data."

    # 3. Validate Data Types and Shapes
    n_train = len(train_data["y"])
    print(f"Train samples: {n_train}")

    # Lexical (Sparse TF-IDF)
    assert sp.issparse(train_data["X_lexical"]), "X_lexical should be sparse (CSR)."
    assert train_data["X_lexical"].shape[0] == n_train, "X_lexical row count mismatch."

    # Behavioral (Sparse CountVec)
    assert sp.issparse(
        train_data["X_behavioral"]
    ), "X_behavioral should be sparse (CSR)."
    assert (
        train_data["X_behavioral"].shape[0] == n_train
    ), "X_behavioral row count mismatch."

    # Semantic (Dense Embeddings)
    assert isinstance(
        train_data["X_semantic"], np.ndarray
    ), "X_semantic should be dense numpy array."
    assert (
        train_data["X_semantic"].shape[0] == n_train
    ), "X_semantic row count mismatch."

    # Meta (Dense Scaled)
    assert isinstance(
        train_data["X_meta"], np.ndarray
    ), "X_meta should be dense numpy array."
    assert train_data["X_meta"].shape[0] == n_train, "X_meta row count mismatch."

    # 4. Check Cache Creation
    cache_files = os.listdir(config.WORKING_DIR)
    print(f"Cache files generated: {len(cache_files)}")
    assert len(cache_files) > 0, "No cache files were saved to the working directory."

    print("Feature Engineering validation passed.")
    return data


def demo_model_factory():
    """
    Demonstrates and validates the ModelFactory class.
    """
    print("\n" + "=" * 40)
    print("DEMO: Model Definitions")
    print("=" * 40)

    factory = ModelFactory()

    # Test each factory method
    models_to_test = [
        ("Lexical Bagger", factory.get_lexical_bagger, "RandomForestClassifier"),
        ("Community Bagger", factory.get_community_bagger, "RandomForestClassifier"),
        ("Semantic Booster", factory.get_semantic_booster, "XGBClassifier"),
        ("Semantic Bagger", factory.get_semantic_bagger, "RandomForestClassifier"),
        ("Metadata Anchor", factory.get_metadata_anchor, "LogisticRegression"),
        ("Temporal Booster", factory.get_temporal_booster, "LGBMClassifier"),
        ("Meta Learner", factory.get_meta_learner, "LogisticRegression"),
    ]

    for name, method, expected_type_name in models_to_test:
        model = method()
        actual_type = type(model).__name__
        print(f"Instantiated {name}: {actual_type}")
        assert (
            actual_type == expected_type_name
        ), f"{name} returned {actual_type}, expected {expected_type_name}"

        # Verify config override took effect (check n_estimators for tree models)
        if hasattr(model, "n_estimators"):
            assert (
                model.n_estimators == 5
            ), f"{name} n_estimators not updated to 5. Got {model.n_estimators}"

    print("Model Factory validation passed.")


def demo_ensemble_pipeline():
    """
    Demonstrates and validates the HexViewEnsemble pipeline.
    """
    print("\n" + "=" * 40)
    print("DEMO: Ensemble Pipeline")
    print("=" * 40)

    # Initialize Ensemble
    ensemble = HexViewEnsemble()

    # --- Step 1: OOF Generation ---
    print("1. Running OOF Generation (Level 1)...")
    oof_preds, y_train = ensemble.train_and_predict_oof(load_cached_data=True)

    # Validation
    n_samples = len(y_train)
    n_models = len(ensemble.base_learners_config)

    print(f"OOF Matrix Shape: {oof_preds.shape}")
    assert oof_preds.shape == (
        n_samples,
        n_models,
    ), f"OOF shape mismatch. Expected ({n_samples}, {n_models}), got {oof_preds.shape}"

    # Check Meta-Learner Saved
    meta_learner_path = os.path.join(ensemble.models_dir, "meta_learner.joblib")
    assert os.path.exists(meta_learner_path), "Meta-learner model file not found."
    print("OOF Generation and Meta-Learner training successful.")

    # --- Step 2: Final Retraining ---
    print("\n2. Running Final Retraining of Base Learners...")
    ensemble.train_final_models(load_cached_data=True)

    # Validation
    for name, _, _ in ensemble.base_learners_config:
        model_path = os.path.join(ensemble.models_dir, f"{name}.joblib")
        assert os.path.exists(model_path), f"Base learner {name} not saved."
    print("All base learners retrained and saved.")

    # --- Step 3: Test Prediction ---
    print("\n3. Generating Test Predictions...")
    ensemble.predict_test(load_cached_data=True)

    # Validation
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found."

    submission_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission Head:\n{submission_df.head()}")

    # Check rows against test metadata
    test_meta = pd.read_parquet(config.TEST_METADATA_PATH)
    assert len(submission_df) == len(
        test_meta
    ), f"Submission row count {len(submission_df)} does not match test set {len(test_meta)}"

    # Check columns
    assert config.ID_COL in submission_df.columns, f"Missing ID column {config.ID_COL}"
    assert (
        config.TARGET_COL in submission_df.columns
    ), f"Missing Target column {config.TARGET_COL}"

    # Check values are probabilities
    probs = submission_df[config.TARGET_COL]
    assert (
        probs.min() >= 0 and probs.max() <= 1
    ), "Predictions are not valid probabilities [0, 1]"

    print("Test Prediction successful.")


if __name__ == "__main__":
    set_seed(42)

    # 1. Setup
    setup_demo_config()

    # 2. Feature Engineering
    # This will create the cache used by subsequent steps
    demo_feature_engineering()

    # 3. Model Definitions
    demo_model_factory()

    # 4. Full Pipeline
    demo_ensemble_pipeline()

    print("\n" + "=" * 40)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 40)
