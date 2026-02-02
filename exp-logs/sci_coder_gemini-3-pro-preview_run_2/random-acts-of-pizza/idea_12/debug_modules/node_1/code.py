import os
import shutil
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_dataset
from library.feature_engineering import prepare_feature_matrices
from library.model_components import DifferentialScaler
from library.execution import train_model, generate_submission

if __name__ == "__main__":
    # Initialize Logger
    logger = setup_logger("demo_execution")
    logger.info("Starting demonstration of library components...")

    # ==========================================
    # 1. Configuration Setup
    # ==========================================
    # Modify Config for a fast, verifiable demonstration run
    logger.info("Configuring environment for fast execution...")

    # Enable Debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples for speed

    # Reduce Cross-Validation complexity
    Config.N_FOLDS = 2

    # Reduce Ensemble complexity
    Config.BAGGING_N_ESTIMATORS = 2

    # Minimize Grid Search space to a single combination
    Config.GRID_SEARCH_PARAMS = {"C": [0.1], "alpha": [1.0], "class_weight": [None]}

    # Ensure clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.ensure_directories()

    # Set random seed for reproducibility
    set_seed(42)

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    logger.info("Step 1: Loading Data...")
    # Force reload from raw files to verify loading logic (ignoring any pre-existing cache)
    df_train, df_val, df_test = load_dataset(load_cached_data=False)

    # Verification
    assert (
        len(df_train) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train set size mismatch: {len(df_train)}"
    assert (
        len(df_val) == Config.DEBUG_SAMPLE_SIZE
    ), f"Val set size mismatch: {len(df_val)}"
    assert (
        len(df_test) == Config.DEBUG_SAMPLE_SIZE
    ), f"Test set size mismatch: {len(df_test)}"
    assert (
        Config.TARGET_COL in df_train.columns
    ), "Target column missing in training data."
    logger.info("Data loading verified successfully.")

    # ==========================================
    # 3. Feature Engineering Demonstration
    # ==========================================
    logger.info("Step 2: Generating Features (Text Embeddings + Tabular)...")
    # This step generates embeddings and processes tabular data
    X_train, y_train, X_val, y_val, X_test = prepare_feature_matrices(
        df_train, df_val, df_test, load_cached_data=False
    )

    # Verification
    # Check dimensions: Rows should match sample size, Cols should be > embedding dim
    assert X_train.shape[0] == Config.DEBUG_SAMPLE_SIZE
    assert X_train.shape[1] > Config.EMBEDDING_DIM, "Features missing tabular data."
    # Check for NaNs
    assert not np.isnan(X_train).any(), "NaNs detected in training features."
    assert not np.isnan(X_test).any(), "NaNs detected in test features."
    logger.info(f"Feature generation verified. Matrix Shape: {X_train.shape}")

    # ==========================================
    # 4. Component Logic Verification
    # ==========================================
    logger.info("Step 3: Verifying DifferentialScaler Logic...")
    # Create synthetic data: 5 samples, 4 embedding dims, 2 tabular dims
    dummy_embed_dim = 4
    dummy_tabular_dim = 2
    dummy_X = np.random.rand(5, dummy_embed_dim + dummy_tabular_dim)

    # Initialize scaler with alpha=10.0
    scaler = DifferentialScaler(alpha=10.0, tabular_start_idx=dummy_embed_dim)
    scaler.fit(dummy_X)
    dummy_X_trans = scaler.transform(dummy_X)

    # Assertions
    # Text part (first 4 columns) should be EXACTLY the same
    assert np.allclose(
        dummy_X[:, :dummy_embed_dim], dummy_X_trans[:, :dummy_embed_dim]
    ), "DifferentialScaler incorrectly modified text embeddings."

    # Tabular part (last 2 columns) should be DIFFERENT (RankGauss + Scaling)
    assert not np.allclose(
        dummy_X[:, dummy_embed_dim:], dummy_X_trans[:, dummy_embed_dim:]
    ), "DifferentialScaler failed to modify tabular features."
    logger.info("DifferentialScaler logic verified.")

    # ==========================================
    # 5. Training Pipeline Execution
    # ==========================================
    logger.info("Step 4: Training Models (Stratified CV)...")
    # Combine train and val for the execution function (it performs its own CV split)
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    # Train models
    fold_models = train_model(X_full, y_full)

    # Verification
    assert (
        len(fold_models) == Config.N_FOLDS
    ), f"Expected {Config.N_FOLDS} trained models, got {len(fold_models)}."
    logger.info("Model training verified.")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    logger.info("Step 5: Generating Submission...")
    generate_submission(fold_models, X_test, df_test)

    # Verification
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Load submission to check format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(sub_df) == Config.DEBUG_SAMPLE_SIZE, "Submission row count mismatch."
    assert list(sub_df.columns) == [
        Config.ID_COL,
        Config.TARGET_COL,
    ], "Submission columns mismatch."

    # Check probability values range
    preds = sub_df[Config.TARGET_COL]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]."

    logger.info(f"Submission verified. Saved to {Config.SUBMISSION_PATH}")
    logger.info("All demonstration steps completed successfully.")
