import os
import sys
import numpy as np
import pandas as pd
import shutil
from sklearn.pipeline import Pipeline

# Import Config first to override settings for the demonstration
from library.config import Config

# ==========================================
# 1. Configuration Overrides for Speed/Demo
# ==========================================
print("Configuring environment for rapid demonstration...")

# Enable Debug mode to slice datasets to a small size
Config.DEBUG = True
Config.DEBUG_SIZE = 20  # Use only 20 samples per split

# Reduce computational load for Cross-Validation
Config.N_FOLDS = 2  # Minimum folds
Config.N_ESTIMATORS = 2  # Minimal bagging estimators
Config.LR_C_RANGE = [1.0]  # Single hyperparameter to avoid extensive GridSearch
Config.LR_CLASS_WEIGHTS = [None]  # Single class weight option

# Redirect outputs to a demo directory to avoid overwriting real work
Config.WORKING_DIR = "./working/demo_execution"
Config.SUBMISSION_DIR = "./working/demo_submission"
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Clean up demo directories if they exist to ensure a fresh run
if os.path.exists(Config.WORKING_DIR):
    shutil.rmtree(Config.WORKING_DIR)
if os.path.exists(Config.SUBMISSION_DIR):
    shutil.rmtree(Config.SUBMISSION_DIR)

os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# Import remaining modules after config setup
from library.utils import set_seed, get_logger
from library.data_manager import load_dataset
from library.feature_extractor import generate_sbert_embeddings
from library.pipeline_manager import LPADFPipelineManager
from library.trainer import run_cross_validation

if __name__ == "__main__":
    # ==========================================
    # 2. Verify Utilities
    # ==========================================
    print("\n--- Verifying Utilities ---")
    set_seed(42)
    logger = get_logger("DemoScript")
    logger.info("Logger and Seeding initialized successfully.")

    # ==========================================
    # 3. Verify Data Manager
    # ==========================================
    print("\n--- Verifying Data Manager ---")
    # Load data with caching disabled to force processing logic execution
    # Debug flag in Config will ensure we only get 20 rows
    df_train, df_val, df_test = load_dataset(load_cached_data=False)

    # Validation
    assert len(df_train) == Config.DEBUG_SIZE, f"Train size mismatch: {len(df_train)}"
    assert len(df_val) == Config.DEBUG_SIZE, f"Val size mismatch: {len(df_val)}"
    assert len(df_test) == Config.DEBUG_SIZE, f"Test size mismatch: {len(df_test)}"

    # Check essential columns
    expected_cols = ["request_id", "text_combined", "subreddit_string"]
    for col in expected_cols:
        assert col in df_train.columns, f"Missing column {col} in train"

    assert "requester_received_pizza" in df_train.columns, "Target missing in train"
    # Test set should not have target (or we don't rely on it)
    logger.info(f"Data Loaded Successfully. Train Shape: {df_train.shape}")

    # ==========================================
    # 4. Verify Feature Extractor
    # ==========================================
    print("\n--- Verifying Feature Extractor ---")
    # Generate embeddings for the sliced dataframes
    train_emb, val_emb, test_emb = generate_sbert_embeddings(
        df_train, df_val, df_test, load_cached_data=False
    )

    # Validation
    # SBERT MiniLM-L6-v2 output dimension is 384
    expected_dim = 384
    assert train_emb.shape == (
        Config.DEBUG_SIZE,
        expected_dim,
    ), f"Train embedding shape error: {train_emb.shape}"
    assert val_emb.shape == (
        Config.DEBUG_SIZE,
        expected_dim,
    ), f"Val embedding shape error: {val_emb.shape}"
    assert test_emb.shape == (
        Config.DEBUG_SIZE,
        expected_dim,
    ), f"Test embedding shape error: {test_emb.shape}"

    # Check normalization (L2 norm should be approx 1.0)
    norm_check = np.linalg.norm(train_emb[0])
    assert np.isclose(
        norm_check, 1.0, atol=1e-5
    ), f"Embeddings not normalized. Norm: {norm_check}"

    logger.info("SBERT Embeddings Generated and Verified.")

    # ==========================================
    # 5. Verify Pipeline Manager
    # ==========================================
    print("\n--- Verifying Pipeline Manager ---")
    manager = LPADFPipelineManager()

    # Test Feature Merging
    merged_df = manager.merge_features(df_train, train_emb)

    # Validation
    # Columns = Original Columns + 384 Embedding Columns
    expected_width = df_train.shape[1] + expected_dim
    assert merged_df.shape == (
        Config.DEBUG_SIZE,
        expected_width,
    ), f"Merged DF shape mismatch. Expected ({Config.DEBUG_SIZE}, {expected_width}), got {merged_df.shape}"

    # Test Pipeline Construction
    pipeline = manager.create_pipeline()
    assert isinstance(
        pipeline, Pipeline
    ), "create_pipeline did not return a Scikit-Learn Pipeline"

    # Check if pipeline steps are correct
    step_names = [step[0] for step in pipeline.steps]
    assert "preprocessor" in step_names, "Pipeline missing preprocessor"
    assert "classifier" in step_names, "Pipeline missing classifier"

    logger.info("Pipeline Manager logic verified.")

    # ==========================================
    # 6. Verify End-to-End Trainer
    # ==========================================
    print("\n--- Verifying End-to-End Training (Integration) ---")
    # This function uses the Config overrides we set at the start
    # It will reload data (hitting the cache we just created or re-processing),
    # compute embeddings (hitting cache or re-computing), and run CV.
    run_cross_validation()

    # Validation of Output
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check submission dimensions
    assert (
        len(submission_df) == Config.DEBUG_SIZE
    ), f"Submission rows mismatch. Expected {Config.DEBUG_SIZE}, got {len(submission_df)}"

    # Check submission columns
    assert "request_id" in submission_df.columns, "request_id missing in submission"
    assert (
        "requester_received_pizza" in submission_df.columns
    ), "prediction column missing in submission"

    # Check values are probabilities
    preds = submission_df["requester_received_pizza"]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions are not valid probabilities"

    logger.info(
        f"End-to-End execution successful. Submission generated at {Config.SUBMISSION_PATH}"
    )
    print("\nAll demonstrations and verifications passed successfully.")
