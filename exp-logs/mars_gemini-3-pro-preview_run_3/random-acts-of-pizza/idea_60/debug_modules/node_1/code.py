import os
import shutil
import numpy as np
import pandas as pd
import warnings

# Import library components
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_data
from library.feature_engineering import FeaturePipeline
from library.training_engine import EnsembleTrainer
from library.inference_engine import EnsemblePredictor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def configure_for_demo():
    """
    Modifies the global Config to run in a fast demonstration mode.
    Reduces dataset size, number of folds, and model complexity.
    """
    print("Configuring environment for fast demonstration...")

    # 1. Enable Debug Mode (Reduces data to 100 rows)
    Config.DEBUG = True

    # 2. Redirect output to a demo-specific directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run/submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 3. Reduce Cross-Validation Folds
    Config.N_FOLDS = 2

    # 4. Minimize Model Hyperparameters for Speed
    # Random Forests
    Config.LEXICAL_RF_PARAMS["n_estimators"] = 5
    Config.COMMUNITY_RF_PARAMS["n_estimators"] = 5
    Config.SEMANTIC_RF_PARAMS["n_estimators"] = 5

    # Gradient Boosters (XGBoost / LightGBM)
    Config.SEMANTIC_XGB_PARAMS.update(
        {
            "n_estimators": 5,
            "early_stopping_rounds": None,  # Disable ES for tiny demo data
        }
    )

    Config.SEMANTIC_LGBM_PARAMS.update(
        {"n_estimators": 5, "verbose": -1, "early_stopping_rounds": None}
    )

    Config.METADATA_LGBM_PARAMS.update(
        {"n_estimators": 5, "verbose": -1, "early_stopping_rounds": None}
    )

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Folds: {Config.N_FOLDS}")


def validate_data_loading():
    """
    Demonstrates and validates the data loading process.
    """
    print("\n--- Step 1: Data Loading ---")

    # Load data (force reload to skip any existing cache in default dir)
    union_train_df, test_df = get_data(load_cached_data=False)

    # Assertions
    assert not union_train_df.empty, "Union training dataframe is empty"
    assert not test_df.empty, "Test dataframe is empty"
    assert (
        Config.TARGET_COL in union_train_df.columns
    ), "Target column missing from training data"

    # Verify Debug mode effect
    if Config.DEBUG:
        assert len(union_train_df) <= 100, "Debug mode failed to truncate training data"
        assert len(test_df) <= 100, "Debug mode failed to truncate test data"

    print("Data Loading Validation Passed.")
    return union_train_df, test_df


def validate_feature_engineering(train_df, test_df):
    """
    Demonstrates and validates the feature engineering pipeline.
    """
    print("\n--- Step 2: Feature Engineering ---")

    pipeline = FeaturePipeline()

    # Execute pipeline
    # Note: This will compute TF-IDF, Embeddings, etc.
    data_dict = pipeline.fit_transform(train_df, test_df, load_cached_data=False)

    # Required keys in the output dictionary
    required_keys = [
        "y_train",
        "X_meta_train",
        "X_meta_test",
        "X_lex_train",
        "X_lex_test",
        "X_beh_train",
        "X_beh_test",
        "X_sem_train",
        "X_sem_test",
    ]

    # Validation
    for key in required_keys:
        assert key in data_dict, f"Missing key in feature dictionary: {key}"

    # Check shapes
    n_train = len(train_df)
    n_test = len(test_df)

    assert len(data_dict["y_train"]) == n_train, "Target vector length mismatch"
    assert data_dict["X_meta_train"].shape[0] == n_train, "Meta train rows mismatch"
    assert data_dict["X_meta_test"].shape[0] == n_test, "Meta test rows mismatch"

    # Check Semantic Embeddings (Dense)
    assert (
        data_dict["X_sem_train"].shape[1] == 384
    ), "Unexpected embedding dimension (expected 384 for all-MiniLM-L6-v2)"

    print("Feature Engineering Validation Passed.")
    return data_dict


def validate_training(data_dict):
    """
    Demonstrates and validates the ensemble training process.
    """
    print("\n--- Step 3: Ensemble Training ---")

    trainer = EnsembleTrainer(data_dict)
    trainer.train_ensemble()

    # Verify Model Artifacts
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    assert os.path.exists(models_dir), "Models directory not created"

    # Check for specific model files
    # We expect fold models (0 and 1 since N_FOLDS=2) and full models
    expected_artifacts = [
        "meta_learner.joblib",
        "lexical_bagger_full.joblib",
        "semantic_booster_fold_0.joblib",
        "semantic_booster_fold_1.joblib",
    ]

    for artifact in expected_artifacts:
        path = os.path.join(models_dir, artifact)
        assert os.path.exists(path), f"Missing model artifact: {artifact}"

    # Check OOF predictions
    oof_path = os.path.join(Config.WORKING_DIR, "oof_predictions.csv")
    assert os.path.exists(oof_path), "OOF predictions file not found"

    print("Training Validation Passed.")


def validate_inference(data_dict):
    """
    Demonstrates and validates the inference and submission generation.
    """
    print("\n--- Step 4: Inference & Submission ---")

    predictor = EnsemblePredictor(data_dict)

    # Run prediction and generation
    predictor.generate_submission()

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file not generated"

    df_sub = pd.read_csv(submission_path)

    # Check columns
    assert (
        Config.ID_COL in df_sub.columns
    ), f"Submission missing ID column: {Config.ID_COL}"
    assert (
        Config.TARGET_COL in df_sub.columns
    ), f"Submission missing Target column: {Config.TARGET_COL}"

    # Check values
    assert df_sub[Config.TARGET_COL].min() >= 0.0, "Probabilities < 0 found"
    assert df_sub[Config.TARGET_COL].max() <= 1.0, "Probabilities > 1 found"

    # Check length (should match debug test size)
    # We need to reload test metadata to know exact size if we didn't pass it,
    # but we know from previous steps.
    assert len(df_sub) > 0, "Submission file is empty"

    print("Inference Validation Passed.")
    print(f"Final Submission generated at: {submission_path}")
    print(df_sub.head())


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # 1. Setup
    configure_for_demo()

    # 2. Data Loading
    train_df, test_df = validate_data_loading()

    # 3. Feature Engineering
    data_dict = validate_feature_engineering(train_df, test_df)

    # 4. Training
    validate_training(data_dict)

    # 5. Inference
    validate_inference(data_dict)

    print("\nAll demonstration steps completed successfully.")
