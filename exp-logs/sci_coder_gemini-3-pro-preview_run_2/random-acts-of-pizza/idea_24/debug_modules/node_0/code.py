import os
import sys
import shutil
import numpy as np
import pandas as pd
import logging
import warnings

# Import library components
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_dataset
from library.feature_extractors import (
    TextEmbedder,
    BayesianSubredditEncoder,
    RankGaussScaler,
)
from library.models import TunedLogisticRegression
from library.trainer import CrossValidationStacker


def main():
    # ==========================================
    # 1. Setup and Configuration Overrides
    # ==========================================
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Configure logger
    logger = setup_logger("demo_execution")
    logger.info("Starting demo execution...")

    # Override Config for speed and isolation
    logger.info("Overriding configuration for fast debug run...")
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples per split
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update cache paths to point to the new working directory
    Config.CACHE_TRAIN_FEATURES = os.path.join(
        Config.WORKING_DIR, "train_merged.parquet"
    )
    Config.CACHE_VAL_FEATURES = os.path.join(Config.WORKING_DIR, "val_merged.parquet")
    Config.CACHE_TEST_FEATURES = os.path.join(Config.WORKING_DIR, "test_merged.parquet")

    # Simplify hyperparameter grids to avoid time-consuming search
    Config.TEXT_EXPERT_GRID = {
        "C": [1.0],
        "solver": ["liblinear"],
        "random_state": [Config.RANDOM_SEED],
    }
    Config.META_LEARNER_GRID = {
        "C": [1.0],
        "solver": ["lbfgs"],
        "random_state": [Config.RANDOM_SEED],
    }

    # Ensure directories exist
    Config.setup()
    set_seed(Config.RANDOM_SEED)

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    logger.info("Step 2: Testing Data Loader...")
    # Force reload from raw data to verify parsing logic
    df_train, df_val, df_test = load_dataset(load_cached_data=False)

    # Validation
    assert (
        len(df_train) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train size mismatch: {len(df_train)}"
    assert len(df_val) == Config.DEBUG_SAMPLE_SIZE, f"Val size mismatch: {len(df_val)}"
    assert (
        len(df_test) == Config.DEBUG_SAMPLE_SIZE
    ), f"Test size mismatch: {len(df_test)}"
    assert Config.TARGET_COL in df_train.columns, "Target column missing in train"
    assert Config.ID_COL in df_test.columns, "ID column missing in test"

    logger.info(f"Data Loaded Successfully. Train shape: {df_train.shape}")

    # ==========================================
    # 3. Feature Extractors Demonstration
    # ==========================================
    logger.info("Step 3: Testing Feature Extractors...")

    # --- 3a. Text Embedder (SBERT) ---
    logger.info("  > Testing TextEmbedder...")
    embedder = TextEmbedder(batch_size=16)
    # Use a specific cache name for this test
    embeddings = embedder.transform(
        df_train, cache_name="train_demo_subset", load_cached_data=False
    )

    # MiniLM-L6-v2 produces 384-dimensional embeddings
    assert embeddings.shape == (
        len(df_train),
        384,
    ), f"Embedding shape mismatch: {embeddings.shape}"
    # Check L2 normalization (norms should be approx 1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings are not L2 normalized"
    logger.info("    TextEmbedder passed.")

    # --- 3b. Bayesian Subreddit Encoder ---
    logger.info("  > Testing BayesianSubredditEncoder...")
    history_encoder = BayesianSubredditEncoder()

    # Fit on training data
    y_train = df_train[Config.TARGET_COL]
    history_encoder.fit(df_train, y_train)

    # Transform
    hist_scores = history_encoder.transform(df_train)

    assert hist_scores.shape == (len(df_train), 1), "History score shape mismatch"
    assert (
        hist_scores.min() >= 0.0 and hist_scores.max() <= 1.0
    ), "History scores out of probability bounds"
    logger.info("    BayesianSubredditEncoder passed.")

    # --- 3c. RankGauss Scaler ---
    logger.info("  > Testing RankGaussScaler...")
    scaler = RankGaussScaler()
    scaler.fit(df_train)
    scaled_meta = scaler.transform(df_train)

    assert scaled_meta.shape == (
        len(df_train),
        len(Config.NUMERIC_COLS),
    ), "Scaled metadata shape mismatch"
    logger.info("    RankGaussScaler passed.")

    # ==========================================
    # 4. Model Wrapper Demonstration
    # ==========================================
    logger.info("Step 4: Testing TunedLogisticRegression...")

    # Create synthetic data for quick model verification
    X_synth = np.random.rand(len(df_train), 10)
    y_synth = y_train.values

    # Initialize model with reduced grid
    model = TunedLogisticRegression(
        param_grid={"C": [0.1, 1.0]},
        cv=2,  # Minimal CV for speed
        random_state=Config.RANDOM_SEED,
    )

    model.fit(X_synth, y_synth)
    preds = model.predict_proba(X_synth)[:, 1]

    assert len(preds) == len(df_train), "Prediction length mismatch"
    assert hasattr(model, "best_params_"), "Model failed to store best params"
    logger.info(f"    Model passed. Best params: {model.best_params_}")

    # ==========================================
    # 5. Full Pipeline Execution (Trainer)
    # ==========================================
    logger.info("Step 5: Testing CrossValidationStacker (Full Pipeline)...")

    stacker = CrossValidationStacker()
    # Override n_folds for demo speed (2 folds instead of 5)
    stacker.n_folds = 2

    # Run the full CV pipeline
    # This handles feature extraction (again), stacking, and submission generation
    auc_score = stacker.run_cv(df_train, df_val, df_test)

    # Verify outputs
    assert isinstance(auc_score, float), "AUC score is not a float"
    assert 0.0 <= auc_score <= 1.0, "AUC score out of bounds"
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Validate submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(sub_df.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Submission columns incorrect"
    assert len(sub_df) == len(df_test), "Submission row count mismatch"

    logger.info(f"Pipeline execution successful. OOF AUC: {auc_score:.4f}")
    logger.info(f"Submission saved at: {Config.SUBMISSION_PATH}")
    logger.info("Demo complete.")


if __name__ == "__main__":
    main()
