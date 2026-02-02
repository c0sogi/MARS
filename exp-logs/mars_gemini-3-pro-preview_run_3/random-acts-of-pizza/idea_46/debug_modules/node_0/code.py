import os
import sys
import shutil
import numpy as np
import pandas as pd

# Import library modules
from library.config import Config
from library.utils import set_seed, log_info
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngineer
from library.ensemble_trainer import EnsembleTrainer


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    log_info("Initializing demonstration script...")

    # Monkey-patch Config for speed optimization (Fast Run Mode)
    # We modify the class attributes directly to affect all modules
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Small sample for quick demonstration
    Config.N_FOLDS = 3  # Reduce folds from 5 to 3

    # Reduce computational cost of Base Learners
    Config.L1_LEXICAL_RF_PARAMS["n_estimators"] = 10
    Config.L1_COMMUNITY_RF_PARAMS["n_estimators"] = 10
    Config.L1_SEMANTIC_XGB_PARAMS["n_estimators"] = 10
    Config.L1_SEMANTIC_RF_PARAMS["n_estimators"] = 10
    Config.L1_META_LGBM_PARAMS["n_estimators"] = 10

    # Reduce Text Vectorization cost
    Config.TFIDF_TEXT_PARAMS["max_features"] = 500
    Config.TFIDF_SUBREDDIT_PARAMS["max_features"] = 100

    # Ensure clean slate for working directory in this demo run
    # (Optional, but good for ensuring load_from_cache=False logic is clear)
    if os.path.exists(Config.CACHE_DIR):
        log_info(f"Cleaning cache directory: {Config.CACHE_DIR}")
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    set_seed(Config.RANDOM_SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    log_info("Step 1: Loading Data...")
    loader = DataLoader()

    # We set load_from_cache=False to force the DataLoader to read raw metadata
    # and apply our Config.DEBUG sampling logic.
    train_df = loader.load_dataset("train", load_from_cache=False)
    val_df = loader.load_dataset("val", load_from_cache=False)
    test_df = loader.load_dataset("test", load_from_cache=False)

    # Verification
    assert (
        len(train_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train set size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(train_df)}"
    assert Config.TARGET_COL in train_df.columns, "Target column missing in Train."
    assert "text_full" in train_df.columns, "Preprocessing failed: 'text_full' missing."
    log_info(f"Data Loaded. Train Shape: {train_df.shape}, Val Shape: {val_df.shape}")

    # -------------------------------------------------------------------------
    # 3. Feature Engineering
    # -------------------------------------------------------------------------
    log_info("Step 2: Generating Feature Views...")
    engineer = FeatureEngineer()

    # Generate all views
    # We disable caching here to ensure we generate features for our specific debug sample
    X_lex_train, X_lex_val, X_lex_test = engineer.get_lexical_view(
        train_df, val_df, test_df, load_from_cache=False
    )
    X_beh_train, X_beh_val, X_beh_test = engineer.get_behavioral_view(
        train_df, val_df, test_df, load_from_cache=False
    )
    X_sem_train, X_sem_val, X_sem_test = engineer.get_semantic_view(
        train_df, val_df, test_df, load_from_cache=False
    )
    X_meta_train, X_meta_val, X_meta_test = engineer.get_metadata_view(
        train_df, val_df, test_df, load_from_cache=False
    )

    # Organize into a dictionary structure for the Trainer
    feature_data = {
        "lexical": {"train": X_lex_train, "val": X_lex_val, "test": X_lex_test},
        "behavioral": {"train": X_beh_train, "val": X_beh_val, "test": X_beh_test},
        "semantic": {"train": X_sem_train, "val": X_sem_val, "test": X_sem_test},
        "metadata": {"train": X_meta_train, "val": X_meta_val, "test": X_meta_test},
    }

    # Verification
    assert X_lex_train.shape[0] == len(train_df), "Lexical feature row count mismatch."
    assert X_meta_train.shape[1] == len(
        Config.METADATA_DENSE_FEATURES
    ), "Metadata feature count mismatch."
    log_info("Feature Engineering Complete.")

    # -------------------------------------------------------------------------
    # 4. Ensemble Training
    # -------------------------------------------------------------------------
    log_info("Step 3: Training Ensemble...")
    trainer = EnsembleTrainer()

    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values

    # A. Generate OOF Predictions (Level 1)
    oof_preds = trainer.get_oof_predictions(feature_data, y_train)
    assert oof_preds.shape == (
        len(train_df),
        len(trainer.base_models),
    ), "OOF shape mismatch."

    # B. Train Meta-Learner (Level 2)
    trainer.train_meta_learner(oof_preds, y_train)

    # C. Retrain Base Models (Validation-Guided)
    trainer.retrain_final_models(feature_data, y_train, y_val)

    # Verification
    assert len(trainer.trained_base_models) == len(
        trainer.base_models
    ), "Not all base models were retrained."

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    log_info("Step 4: Generating Submission...")

    test_ids = test_df[Config.ID_COL].values
    submission_df = trainer.generate_submission(feature_data, test_ids)

    # Verification
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."
    assert submission_df.shape == (
        len(test_df),
        2,
    ), "Submission DataFrame has incorrect shape."
    assert list(submission_df.columns) == [
        Config.ID_COL,
        Config.TARGET_COL,
    ], "Submission columns are incorrect."

    # Check for NaNs
    assert (
        not submission_df[Config.TARGET_COL].isnull().any()
    ), "Submission contains NaNs."

    log_info(f"Submission generated successfully at {Config.SUBMISSION_PATH}")
    print(submission_df.head())

    log_info("Demonstration completed successfully.")


if __name__ == "__main__":
    main()
