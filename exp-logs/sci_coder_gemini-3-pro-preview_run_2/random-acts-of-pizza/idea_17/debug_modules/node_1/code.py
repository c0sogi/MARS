import sys
import os
import shutil
import numpy as np
import pandas as pd
import torch
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import DataLoader
from library.feature_engineering import Embedder, ViewTransformer
from library.model_builder import ModelBuilder
from library.trainer import Trainer
from library.predictor import Predictor

# ==========================================
# Configuration & Setup
# ==========================================


def setup_demo_config():
    """
    Overrides the default configuration to run a fast demo:
    - Uses a separate working directory.
    - Reduces the number of folds and estimators.
    - Simplifies grid search.
    - Defines distinct cache paths for the demo.
    """
    print("Setting up demo configuration...")

    # Use a separate working directory for the demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Clean up previous demo run artifacts to prevent stale cache issues (Cite debug_lesson_4)
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Override Cache Paths for Tabular Data
    Config.TRAIN_TABULAR_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_TABULAR_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_TABULAR_PATH = os.path.join(Config.WORKING_DIR, "test_features.parquet")

    # Override Cache Paths for Embeddings (View 1: Request)
    Config.TRAIN_REQ_EMB_PATH = os.path.join(Config.WORKING_DIR, "train_req.npy")
    Config.VAL_REQ_EMB_PATH = os.path.join(Config.WORKING_DIR, "val_req.npy")
    Config.TEST_REQ_EMB_PATH = os.path.join(Config.WORKING_DIR, "test_req.npy")

    # Override Cache Paths for Embeddings (View 2: History)
    Config.TRAIN_HIST_EMB_PATH = os.path.join(Config.WORKING_DIR, "train_hist.npy")
    Config.VAL_HIST_EMB_PATH = os.path.join(Config.WORKING_DIR, "val_hist.npy")
    Config.TEST_HIST_EMB_PATH = os.path.join(Config.WORKING_DIR, "test_hist.npy")

    # Reduce Complexity for Speed
    Config.N_FOLDS = 2
    Config.BAGGING_N_ESTIMATORS = 2
    Config.HISTORY_PCA_COMPONENTS = 5  # Reduced dimensionality

    # Simplify Grid Search (No search, just one config)
    Config.GRID_SEARCH_PARAMS = {"C": [1.0], "class_weight": [None]}

    # Disable tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================
# Monkey Patching for Data Subsetting
# ==========================================

# Save reference to the original method
original_load_dataset = DataLoader.load_dataset


def mocked_load_dataset(
    self, split: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Wraps the original data loading logic but slices the result to a small subset
    to ensure the demo runs quickly.
    """
    # Call original method. We allow it to cache the FULL dataset to parquet
    # (which is fast to load subsequently), but we always return a slice.
    df = original_load_dataset(self, split, load_cached_data=load_cached_data)

    # Slice to 50 samples for the demo
    n_samples = 50
    if len(df) > n_samples:
        # print(f"Mock DataLoader: Slicing {split} data to {n_samples} samples.")
        df = df.iloc[:n_samples].reset_index(drop=True)

    return df


# Apply the patch
DataLoader.load_dataset = mocked_load_dataset


# ==========================================
# Main Execution
# ==========================================


def main():
    # 1. Initialize
    setup_demo_config()
    set_seed(42)
    logger = setup_logger("DemoScript")

    logger.info("Starting AMBLE Demo Execution...")

    # ==========================================
    # Step 1: Verify Data Loading & Feature Engineering
    # ==========================================
    logger.info("\n--- Step 1: Testing Data Loading & Embedding ---")

    dl = DataLoader()
    # Load train data (mocked to return 50 rows)
    df_train = dl.load_dataset("train", load_cached_data=False)

    # Assertions
    assert len(df_train) == 50, f"Expected 50 training samples, got {len(df_train)}"
    assert "text_view" in df_train.columns, "Missing 'text_view' column"
    assert "history_view" in df_train.columns, "Missing 'history_view' column"
    assert "requester_received_pizza" in df_train.columns, "Missing target column"

    # Verify Embedder
    embedder = Embedder()
    # Force computation (load_cached_data=False) to verify SBERT encoding works
    logger.info("Generating embeddings for subset...")
    embeddings = embedder.get_embeddings(
        df_train, "train", "request", load_cached_data=False
    )

    assert embeddings.shape == (
        50,
        384,
    ), f"Expected embedding shape (50, 384), got {embeddings.shape}"
    assert not np.isnan(embeddings).any(), "Embeddings contain NaNs"
    logger.info("Data Loading and Embedding generation verified.")

    # ==========================================
    # Step 2: Verify ViewTransformer
    # ==========================================
    logger.info("\n--- Step 2: Testing ViewTransformer ---")
    vt = ViewTransformer()

    # Create dummy data matching expected shapes
    X_req_dummy = np.random.rand(50, 384)
    X_hist_dummy = np.random.rand(50, 384)
    X_meta_dummy = np.random.rand(50, 10)  # 10 numerical features

    # Test Fit
    vt.fit(X_hist_dummy, X_meta_dummy)
    assert vt.is_fitted, "ViewTransformer should be fitted after calling fit()."

    # Test Transform
    X_fused = vt.transform(X_req_dummy, X_hist_dummy, X_meta_dummy)

    # Expected dimensions:
    # Request (384) + History PCA (5) + Metadata (10) = 399
    expected_dim = 384 + Config.HISTORY_PCA_COMPONENTS + 10
    assert X_fused.shape == (
        50,
        expected_dim,
    ), f"Expected fused shape (50, {expected_dim}), got {X_fused.shape}"
    logger.info("ViewTransformer logic verified.")

    # ==========================================
    # Step 3: Verify Model Building
    # ==========================================
    logger.info("\n--- Step 3: Testing ModelBuilder ---")
    mb = ModelBuilder()
    model = mb.get_bagged_ensemble(C=1.0)

    assert (
        model.n_estimators == Config.BAGGING_N_ESTIMATORS
    ), "Model n_estimators mismatch with Config"
    assert model.estimator.C == 1.0, "Base estimator hyperparameter mismatch"
    logger.info("ModelBuilder verified.")

    # ==========================================
    # Step 4: Full Training Pipeline
    # ==========================================
    logger.info("\n--- Step 4: Running Trainer ---")
    trainer = Trainer()

    # Run training (uses mocked DataLoader, so it trains on 50 samples)
    try:
        trainer.run_training()
        logger.info("Training pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e

    # Verify Model Artifacts
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(
            Config.WORKING_DIR, "models", f"model_fold_{fold}.joblib"
        )
        trans_path = os.path.join(
            Config.WORKING_DIR, "models", f"transformer_fold_{fold}.joblib"
        )
        assert os.path.exists(model_path), f"Model for fold {fold} missing."
        assert os.path.exists(trans_path), f"Transformer for fold {fold} missing."

    # ==========================================
    # Step 5: Inference Pipeline
    # ==========================================
    logger.info("\n--- Step 5: Running Predictor ---")
    predictor = Predictor()

    # Run inference (uses mocked DataLoader for test set)
    try:
        predictor.generate_submission()
        logger.info("Inference pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise e

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Checks
    assert len(df_sub) == 50, f"Expected 50 predictions, got {len(df_sub)}"
    assert "request_id" in df_sub.columns, "Missing request_id in submission"
    assert "requester_received_pizza" in df_sub.columns, "Missing target in submission"

    # Check probability range
    probs = df_sub["requester_received_pizza"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of [0, 1] range"

    logger.info("\nAll demonstration steps completed successfully!")
    print("\nSample Submission Output:")
    print(df_sub.head())


if __name__ == "__main__":
    main()
