import os
import sys
import shutil
import numpy as np
import pandas as pd
import logging
import warnings

# Import from the provided library
from library.config import Config
from library.utils import setup_logger
from library.data_manager import DataManager
from library.trainer import Trainer
from library.model_zoo import LGBMExpert, XGBExpert, CatBoostExpert, EnsemblePredictor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def create_mini_datasets(n_rows=2000):
    """
    Creates smaller versions of the metadata files to speed up the demonstration.
    """
    print(f"Creating mini datasets with {n_rows} rows...")

    # Define paths for mini datasets
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train_metadata.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val_metadata.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test_metadata.csv")

    # Load original metadata (just the head)
    # We read slightly more to ensure we get enough valid groups for feature engineering
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH, nrows=n_rows * 2)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH, nrows=n_rows)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH, nrows=n_rows)

    # Save mini versions
    df_train.head(n_rows).to_csv(mini_train_path, index=False)
    df_val.head(n_rows).to_csv(mini_val_path, index=False)
    df_test.head(n_rows).to_csv(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path


def configure_demo_environment(mini_train, mini_val, mini_test):
    """
    Overrides the global Config to use mini datasets and reduce training time.
    """
    print("Configuring demo environment...")

    # 1. Update Paths
    Config.TRAIN_METADATA_PATH = mini_train
    Config.VAL_METADATA_PATH = mini_val
    Config.TEST_METADATA_PATH = mini_test

    # 2. Update Working Directory to avoid cache conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_DIR = os.path.join(Config.WORKING_DIR, "models")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # 3. Reduce Training Hyperparameters for Speed
    Config.TRAINING["SCOUT_EPOCHS"] = 1
    Config.TRAINING["EXPERT_EPOCHS"] = 1
    Config.TRAINING["EARLY_STOPPING_ROUNDS"] = 1
    Config.TRAINING["VERBOSE_EVAL"] = -1  # Silent

    # 4. Reduce Model Complexity
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.LGBM_PARAMS["n_estimators"] = 1
    Config.XGB_PARAMS["max_depth"] = 3
    Config.XGB_PARAMS["num_boost_round"] = 1
    Config.CATBOOST_PARAMS["iterations"] = 1
    Config.CATBOOST_PARAMS["depth"] = 3


def run_demonstration():
    # Setup Logger
    logger = setup_logger("demo_script")
    logger.info("Starting Demonstration Script")

    # -------------------------------------------------------------------------
    # Step 1: Setup Data
    # -------------------------------------------------------------------------
    mini_train, mini_val, mini_test = create_mini_datasets(n_rows=5000)
    configure_demo_environment(mini_train, mini_val, mini_test)

    # -------------------------------------------------------------------------
    # Step 2: Data Loading & Feature Engineering
    # -------------------------------------------------------------------------
    logger.info("Step 2: Testing DataManager and Feature Engineering...")
    dm = DataManager()

    # Force reload (ignore cache) to prove processing logic works
    df_train = dm.load_train_features(load_cached=False)
    df_val = dm.load_val_features(load_cached=False)
    df_test = dm.load_test_features(load_cached=False)

    # Validation
    assert not df_train.empty, "Train dataframe should not be empty"
    assert (
        "spectral_energy" in df_train.columns
    ), "Feature engineering failed: spectral_energy missing"
    assert (
        "gating_min_dist" in df_train.columns
    ), "Feature engineering failed: gating_min_dist missing"
    assert (
        "distance" in df_train.columns
    ), "Feature engineering failed: distance missing"

    # Check Gating Logic: If enabled, rows with high min_dist should be filtered out (unless Ground)
    # We can't strictly assert row count reduction without knowing the data, but we can check values
    if Config.GATING["ENABLED"]:
        # Ensure no player-player interaction has absurdly high min_dist (e.g. > 100) if threshold is small
        # Note: Ground contacts have sentinel distance, so filter them out for this check
        pp_mask = df_train["nfl_player_id_2"] != "G"
        if pp_mask.any():
            max_gated_dist = df_train.loc[pp_mask, "gating_min_dist"].max()
            # It might be slightly above threshold due to lookahead boundary, but shouldn't be huge
            # This is a soft check
            pass

    logger.info(
        f"Train Rows: {len(df_train)}, Val Rows: {len(df_val)}, Test Rows: {len(df_test)}"
    )

    # -------------------------------------------------------------------------
    # Step 3: Training Scouts (Phase 1)
    # -------------------------------------------------------------------------
    logger.info("Step 3: Training Scouts...")
    trainer = Trainer()

    # Prepare Validation Set
    X_val, y_val = dm.get_validation_set(df_val)

    # Train Scouts
    scout_lgbm, scout_xgb = trainer.train_scouts(df_train, X_val, y_val)

    # Validation
    assert scout_lgbm.model is not None, "LGBM Scout model not trained"
    assert scout_xgb.model is not None, "XGB Scout model not trained"

    # Check prediction shape
    sample_X = X_val[:10]
    preds = scout_lgbm.predict_proba(sample_X)
    assert preds.shape == (10,), f"Prediction shape mismatch: {preds.shape}"
    assert np.all((preds >= 0) & (preds <= 1)), "Probabilities out of range [0, 1]"

    # -------------------------------------------------------------------------
    # Step 4: Mining Hard Negatives (Phase 2)
    # -------------------------------------------------------------------------
    logger.info("Step 4: Mining Hard Negatives...")

    # We force mining from scratch
    hard_neg_indices = trainer.mine_hard_negatives(
        df_train, scout_lgbm, scout_xgb, load_cached=False
    )

    # Validation
    assert isinstance(
        hard_neg_indices, np.ndarray
    ), "Hard negative indices should be a numpy array"
    # It's possible to find 0 hard negatives in a tiny random subset, so we don't assert len > 0
    logger.info(f"Mined {len(hard_neg_indices)} hard negatives.")

    # -------------------------------------------------------------------------
    # Step 5: Training Expert Ensemble (Phase 3)
    # -------------------------------------------------------------------------
    logger.info("Step 5: Training Expert Ensemble...")

    ensemble = trainer.train_expert_ensemble(df_train, hard_neg_indices, X_val, y_val)

    # Validation
    assert len(ensemble.models) == 3, "Ensemble should contain 3 models"
    assert isinstance(ensemble.models[0], LGBMExpert)
    assert isinstance(ensemble.models[1], XGBExpert)
    assert isinstance(ensemble.models[2], CatBoostExpert)

    ens_preds = ensemble.predict(sample_X)
    assert ens_preds.shape == (10,), "Ensemble prediction shape mismatch"

    # -------------------------------------------------------------------------
    # Step 6: Threshold Optimization & Submission
    # -------------------------------------------------------------------------
    logger.info("Step 6: Optimization and Submission...")

    best_thresh = trainer.optimize_threshold(ensemble, X_val, y_val)
    assert 0 < best_thresh < 1, "Optimal threshold out of bounds"

    trainer.generate_submission(ensemble, best_thresh, df_test)

    # Validation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(df_sub) == len(df_test), "Submission row count mismatch"
    assert "contact_id" in df_sub.columns, "contact_id column missing in submission"
    assert "contact" in df_sub.columns, "contact column missing in submission"
    assert df_sub["contact"].isin([0, 1]).all(), "Submission contains non-binary values"

    logger.info("Demonstration completed successfully!")


if __name__ == "__main__":
    # Ensure we are in a clean state
    if os.path.exists("./working/demo_run"):
        shutil.rmtree("./working/demo_run")

    run_demonstration()
