import os
import sys
import shutil
import warnings
import pandas as pd
import numpy as np
import scipy.sparse as sp

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.data_manager import DataManager
from library.feature_engine import (
    GranularLexicalVectorizer,
    CommunityVectorizer,
    MetadataScaler,
    SemanticEmbedder,
)
from library.training_engine import TrainingEngine
from library.inference_engine import InferenceEngine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def configure_demo_environment():
    """
    Overrides default configuration to ensure the demo runs quickly and
    writes to a specific demo directory.
    """
    print("--- Configuring Demo Environment ---")

    # 1. Redirect Paths to a demo directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Reduce Complexity for Speed
    Config.N_FOLDS = 2

    # Reduce estimators for all ensemble models
    Config.LEXICAL_BAGGER_PARAMS["n_estimators"] = 2
    Config.COMMUNITY_BAGGER_PARAMS["n_estimators"] = 2
    Config.SEMANTIC_BOOSTER_PARAMS["n_estimators"] = 2
    Config.SEMANTIC_GRADIENT_PARAMS["n_estimators"] = 2
    Config.SEMANTIC_BAGGER_PARAMS["n_estimators"] = 2
    Config.TEMPORAL_BOOSTER_PARAMS["n_estimators"] = 2

    # Ensure LightGBM is silent
    Config.SEMANTIC_GRADIENT_PARAMS["verbose"] = -1
    Config.TEMPORAL_BOOSTER_PARAMS["verbose"] = -1

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"N_FOLDS set to: {Config.N_FOLDS}")
    print("Model estimators reduced to 2 for speed.")
    print("-" * 30)


def demo_data_manager():
    """
    Demonstrates loading and processing data using DataManager.
    """
    print("\n--- Demo: DataManager ---")
    dm = DataManager()

    # Load union data (Train + Val) and Test data
    # We force load_cached_data=False to ensure we demonstrate processing logic
    train_df, test_df = dm.load_union_data(load_cached_data=False)

    print(f"Loaded Train Union Shape: {train_df.shape}")
    print(f"Loaded Test Shape: {test_df.shape}")

    # Validation
    assert not train_df.empty, "Training dataframe should not be empty"
    assert not test_df.empty, "Test dataframe should not be empty"
    assert "text_combined" in train_df.columns, "Feature engineering (text) failed"
    assert (
        "subreddit_string" in train_df.columns
    ), "Feature engineering (subreddits) failed"
    assert (
        "requester_received_pizza" in train_df.columns
    ), "Target column missing in train"

    print("DataManager validation passed.")
    return train_df


def demo_feature_engine(sample_df):
    """
    Demonstrates the usage of a Feature Engine component (GranularLexicalVectorizer).
    """
    print("\n--- Demo: Feature Engine (GranularLexicalVectorizer) ---")

    # Take a small sample for speed
    small_df = sample_df.head(50).copy()

    vectorizer = GranularLexicalVectorizer()

    # Fit
    print("Fitting vectorizer...")
    vectorizer.fit(small_df)

    # Transform
    print("Transforming data...")
    X_sparse = vectorizer.transform(small_df)

    print(f"Output Matrix Shape: {X_sparse.shape}")
    print(f"Output Type: {type(X_sparse)}")

    # Validation
    assert sp.issparse(X_sparse), "Output should be a sparse matrix"
    assert X_sparse.shape[0] == 50, "Output rows should match input rows"
    assert vectorizer.is_fitted, "Vectorizer should be marked as fitted"

    print("Feature Engine validation passed.")


def demo_training_engine():
    """
    Demonstrates the TrainingEngine: Feature extraction, CV training, and OOF generation.
    """
    print("\n--- Demo: TrainingEngine ---")

    trainer = TrainingEngine()
    trainer.run()

    # Verify Outputs
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    oof_path = os.path.join(Config.WORKING_DIR, "oof_predictions.csv")

    # Check for OOF file
    assert os.path.exists(oof_path), "OOF predictions file was not created"
    oof_df = pd.read_csv(oof_path)
    assert not oof_df.empty, "OOF file is empty"
    print(f"OOF Predictions generated at: {oof_path}")

    # Check for a few model files (Stable models)
    expected_models = [
        "lexical_bagger_full.joblib",
        "community_bagger_full.joblib",
        "metadata_scaler.joblib",
    ]
    for model_file in expected_models:
        path = os.path.join(models_dir, model_file)
        assert os.path.exists(path), f"Model artifact {model_file} missing"

    print("TrainingEngine validation passed.")


def demo_inference_engine():
    """
    Demonstrates the InferenceEngine: Loading models, generating predictions, and submission.
    """
    print("\n--- Demo: InferenceEngine ---")

    inferencer = InferenceEngine()
    inferencer.run()

    # Verify Submission
    sub_path = Config.SUBMISSION_PATH
    assert os.path.exists(sub_path), "Submission file was not created"

    sub_df = pd.read_csv(sub_path)

    # Basic Submission Checks
    assert "request_id" in sub_df.columns, "Submission missing request_id"
    assert (
        "requester_received_pizza" in sub_df.columns
    ), "Submission missing target column"
    assert len(sub_df) > 0, "Submission file is empty"

    # Check probabilities range
    probs = sub_df["requester_received_pizza"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of [0, 1] range"

    print(f"Submission generated at: {sub_path}")
    print(f"Submission Shape: {sub_df.shape}")
    print("InferenceEngine validation passed.")


if __name__ == "__main__":
    # 1. Set global seed
    set_seed(42)

    # 2. Configure environment for demo
    configure_demo_environment()

    # 3. Run DataManager Demo
    train_df_sample = demo_data_manager()

    # 4. Run Feature Engine Demo
    demo_feature_engine(train_df_sample)

    # 5. Run Training Pipeline
    # This will train models on the full (but small) dataset with reduced estimators
    demo_training_engine()

    # 6. Run Inference Pipeline
    # This will generate the final submission
    demo_inference_engine()

    print("\n=== All Demonstrations Completed Successfully ===")
