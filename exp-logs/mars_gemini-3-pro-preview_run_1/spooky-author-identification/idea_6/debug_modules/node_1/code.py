import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library.configuration import Config
from library import utilities
from library import data_handling
from library import linear_expert
from library import transformer_expert
from library import meta_learner

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Creates a temporary working directory and generates a small subset of the data
    to ensure the demonstration runs quickly.
    """
    print(">>> Setting up demo environment...")

    # Define demo paths
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Load original metadata
    full_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    full_test = pd.read_csv(Config.TEST_DATA_PATH)

    # Sample data for speed (50 train samples, 20 test samples)
    # Ensure we have at least a few samples per class for stratification
    subset_train = full_train.groupby("author", group_keys=False).apply(
        lambda x: x.sample(min(len(x), 20), random_state=Config.SEED)
    )
    subset_train = subset_train.sample(frac=1, random_state=Config.SEED).reset_index(
        drop=True
    )
    subset_test = full_test.sample(n=20, random_state=Config.SEED).reset_index(
        drop=True
    )

    # Save subsets
    train_path = os.path.join(demo_dir, "train_subset.csv")
    test_path = os.path.join(demo_dir, "test_subset.csv")
    subset_train.to_csv(train_path, index=False)
    subset_test.to_csv(test_path, index=False)

    print(f"    Created subset: Train={subset_train.shape}, Test={subset_test.shape}")

    return demo_dir, train_path, test_path


def patch_configuration(demo_dir, train_path, test_path):
    """
    Dynamically overrides Config attributes to optimize for the demo run.
    """
    print(">>> Patching Configuration for speed...")

    # Paths
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_FILE = os.path.join(demo_dir, "submission.csv")
    Config.TRAIN_DATA_PATH = train_path
    Config.TEST_DATA_PATH = test_path

    # Training Parameters
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.NUM_FOLDS = 2  # Run only 2 folds
    Config.TRAIN_BATCH_SIZE = 4  # Small batch size
    Config.VALID_BATCH_SIZE = 8
    Config.MAX_LENGTH = 64  # Short sequence length for speed

    # XGBoost Parameters (Speed up)
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["early_stopping_rounds"] = 5

    # Note: We keep Config.MODEL_NAME as 'microsoft/deberta-v3-large' because
    # the optimizer logic in transformer_expert.py hardcodes layer counts (24).
    # Using a smaller model would mismatch the layer decay logic.
    # The A100 GPU can handle the large model easily even for this demo.


def verify_utilities():
    """
    Demonstrates and verifies utility functions.
    """
    print("\n>>> Verifying Utilities...")

    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)

    # Test Meta-Feature Extraction
    meta_feats = utilities.extract_meta_features(
        df_train, cache_name="demo_train", load_cached_data=False
    )

    # Validation
    assert isinstance(meta_feats, pd.DataFrame)
    assert len(meta_feats) == len(df_train)
    assert "meta_char_len" in meta_feats.columns
    assert "meta_punct_density" in meta_feats.columns

    print("    extract_meta_features: OK")


def verify_data_handling():
    """
    Demonstrates and verifies data handling functions.
    """
    print("\n>>> Verifying Data Handling...")

    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # 1. Test Stratified Folds
    df_folds = data_handling.get_stratified_folds(
        df_train, num_folds=Config.NUM_FOLDS, seed=Config.SEED
    )

    assert "fold" in df_folds.columns
    assert df_folds["fold"].nunique() == Config.NUM_FOLDS
    assert not df_folds["fold"].isnull().any()
    print("    get_stratified_folds: OK")

    # 2. Test TF-IDF Features
    X_train, X_test = data_handling.get_tfidf_features(
        df_train, df_test, load_cached_data=False
    )

    assert X_train.shape[0] == len(df_train)
    assert X_test.shape[0] == len(df_test)
    assert X_train.shape[1] == X_test.shape[1]
    print("    get_tfidf_features: OK")


def run_linear_expert_demo():
    """
    Runs the Linear Expert training pipeline.
    """
    print("\n>>> Running Linear Expert Demo...")

    # Force retraining by setting load_cached_data=False
    oof_preds, test_preds = linear_expert.train_linear_expert(load_cached_data=False)

    # Validation
    n_train = len(pd.read_csv(Config.TRAIN_DATA_PATH))
    n_test = len(pd.read_csv(Config.TEST_DATA_PATH))

    assert oof_preds.shape == (n_train, 3)
    assert test_preds.shape == (n_test, 3)
    assert not np.isnan(oof_preds).any()

    print("    Linear Expert Output Shapes Verified.")
    return oof_preds, test_preds


def run_transformer_expert_demo():
    """
    Runs the Transformer Expert training pipeline.
    """
    print("\n>>> Running Transformer Expert Demo...")

    # Force retraining
    oof_preds, test_preds = transformer_expert.train_transformer_expert(
        load_cached_data=False
    )

    # Validation
    n_train = len(pd.read_csv(Config.TRAIN_DATA_PATH))
    n_test = len(pd.read_csv(Config.TEST_DATA_PATH))

    assert oof_preds.shape == (n_train, 3)
    assert test_preds.shape == (n_test, 3)

    # Check if probabilities sum roughly to 1 (softmax output)
    assert np.allclose(oof_preds.sum(axis=1), 1.0, atol=1e-5)

    print("    Transformer Expert Output Shapes Verified.")
    return oof_preds, test_preds


def run_meta_learner_demo(trans_oof, trans_test, lin_oof, lin_test):
    """
    Runs the Meta Learner (XGBoost) pipeline.
    """
    print("\n>>> Running Meta Learner Demo...")

    submission = meta_learner.train_predict_xgboost(
        transformer_oof=trans_oof,
        transformer_test=trans_test,
        linear_oof=lin_oof,
        linear_test=lin_test,
        save_submission=True,
    )

    # Validation
    n_test = len(pd.read_csv(Config.TEST_DATA_PATH))

    assert isinstance(submission, pd.DataFrame)
    assert len(submission) == n_test
    assert list(submission.columns) == ["id", "EAP", "HPL", "MWS"]

    # Check file on disk
    assert os.path.exists(Config.SUBMISSION_FILE)

    print("    Submission file generated successfully.")
    print(f"    Head of submission:\n{submission.head()}")


if __name__ == "__main__":
    # 1. Initialize Seeds
    utilities.seed_everything(Config.SEED)

    # 2. Setup Environment and Data
    demo_dir, train_path, test_path = setup_demo_environment()

    # 3. Patch Configuration for Speed
    patch_configuration(demo_dir, train_path, test_path)

    # 4. Verify Components
    verify_utilities()
    verify_data_handling()

    # 5. Run Experts
    # Linear Model (TF-IDF + LR)
    lin_oof, lin_test = run_linear_expert_demo()

    # Transformer Model (DeBERTa)
    # This will download the model config/weights if not cached, then train for 1 epoch on subset
    trans_oof, trans_test = run_transformer_expert_demo()

    # 6. Run Meta Learner (Ensemble)
    run_meta_learner_demo(trans_oof, trans_test, lin_oof, lin_test)

    print("\n>>> Demo Completed Successfully.")
