import os
import shutil
import numpy as np
import pandas as pd
import logging
from library.config import Config
from library.utils import seed_everything, setup_logging
from library.features import FeatureEngineer
from library.data_pipeline import DataPipeline
from library.models import TriEnsemble
from library.mining import ScoutMiner
from library.training_flow import TrainingPipeline


# =============================================================================
# 1. Configuration for Fast Demonstration
# =============================================================================
class FastConfig(Config):
    """
    Subclass of Config to override paths and hyperparameters for a fast,
    self-contained demonstration run.
    """

    # Use a specific demo directory in working/
    WORKING_DIR = "./working/demo_run"
    SUBMISSION_DIR = "./working/demo_run"

    # Re-define paths to ensure they point to the demo directory
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "features_train.parquet")
    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "features_val.parquet")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "features_test.parquet")
    CACHE_HARD_NEGATIVES = os.path.join(WORKING_DIR, "hard_negative_indices.npy")
    CACHE_BEST_THRESHOLD = os.path.join(WORKING_DIR, "best_threshold.npy")

    MODEL_SCOUT_LGBM = os.path.join(WORKING_DIR, "models/scouts/lgbm_model.joblib")
    MODEL_SCOUT_XGB = os.path.join(WORKING_DIR, "models/scouts/xgb_model.joblib")
    MODEL_SCOUT_CAT = os.path.join(WORKING_DIR, "models/scouts/cat_model.joblib")

    MODEL_EXPERT_LGBM = os.path.join(WORKING_DIR, "models/experts/lgbm_model.joblib")
    MODEL_EXPERT_XGB = os.path.join(WORKING_DIR, "models/experts/xgb_model.joblib")
    MODEL_EXPERT_CAT = os.path.join(WORKING_DIR, "models/experts/cat_model.joblib")

    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "mini_sample_submission.csv")

    # Enable Debug mode to sample data
    DEBUG = True
    DEBUG_SAMPLE_SIZE = 500  # Small sample for speed

    # Reduce Model Complexity for Speed
    LGBM_PARAMS = Config.LGBM_PARAMS.copy()
    LGBM_PARAMS.update({"n_estimators": 5, "num_leaves": 8, "verbose": -1})

    XGB_PARAMS = Config.XGB_PARAMS.copy()
    XGB_PARAMS.update({"n_estimators": 5, "max_depth": 3, "verbosity": 0})

    CAT_PARAMS = Config.CAT_PARAMS.copy()
    CAT_PARAMS.update({"iterations": 5, "depth": 3, "verbose": 0})


def clean_working_dir():
    if os.path.exists(FastConfig.WORKING_DIR):
        shutil.rmtree(FastConfig.WORKING_DIR)
    os.makedirs(FastConfig.WORKING_DIR, exist_ok=True)


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Setup
    setup_logging(level=logging.INFO)
    seed_everything(FastConfig.SEED)
    clean_working_dir()

    print("\n=== Step 1: Feature Engineering Verification ===")
    # Instantiate FeatureEngineer
    engineer = FeatureEngineer(config=FastConfig)

    # Generate Train Features (Debug Mode)
    # This tests _load_tracking_data, _merge_tracking_at_lag, and _compute_dual_basis_and_project
    df_train = engineer.create_train_features(load_cached_data=False, debug=True)

    # Validation
    assert not df_train.empty, "Feature engineering returned empty DataFrame."
    assert "contact" in df_train.columns, "Target column 'contact' missing."
    assert "dist_lag0" in df_train.columns, "Feature 'dist_lag0' missing."
    assert "v_comp1_lag0" in df_train.columns, "Feature 'v_comp1_lag0' missing."
    print(f"Generated train features shape: {df_train.shape}")
    print("Feature Engineering logic verified.")

    print("\n=== Step 2: Data Pipeline & Scout Dataset Construction ===")
    data_pipeline = DataPipeline(config=FastConfig)

    # Construct Scout Dataset (Balanced 1:1)
    # Note: If random sampling results in 0 positives in the tiny debug set, this might be empty.
    # We'll check if we have positives first.
    n_pos = df_train["contact"].sum()
    if n_pos > 0:
        df_scout = data_pipeline.construct_scout_dataset(df_train)

        # Validation
        scout_pos = df_scout["contact"].sum()
        scout_neg = (df_scout["contact"] == 0).sum()
        assert (
            scout_pos == scout_neg
        ), f"Scout dataset not balanced! Pos: {scout_pos}, Neg: {scout_neg}"
        print(f"Scout dataset created with {len(df_scout)} samples (Balanced).")
    else:
        print("Warning: No positives in debug sample. Skipping balance check.")
        df_scout = df_train.copy()  # Fallback for demo flow

    print("\n=== Step 3: Model Training (Scouts) ===")
    # Instantiate TriEnsemble
    scout_ensemble = TriEnsemble(config=FastConfig)

    # Train only LGBM and XGB for speed (skip CatBoost if not installed or for speed)
    model_keys = ["lgbm", "xgb"]
    scout_ensemble.fit(df_scout, df_val=None, model_names=model_keys)

    # Validation
    probs = scout_ensemble.predict_proba(df_scout)
    assert len(probs) == len(df_scout), "Prediction length mismatch."
    assert (
        probs.min() >= 0 and probs.max() <= 1
    ), "Predictions out of probability bounds."
    print("Scout models trained and prediction verified.")

    print("\n=== Step 4: Hard Negative Mining ===")
    miner = ScoutMiner(config=FastConfig)

    # Mine hard negatives from the original training set
    hard_indices = miner.mine_hard_negatives(
        df_train, scout_ensemble, load_cached_data=False
    )

    # Validation
    assert isinstance(hard_indices, np.ndarray), "Hard indices should be a numpy array."
    # Check that indices actually belong to the dataframe
    if len(hard_indices) > 0:
        assert np.isin(
            hard_indices, df_train.index
        ).all(), "Mined indices not found in source DataFrame."
    print(f"Mined {len(hard_indices)} hard negatives.")

    print("\n=== Step 5: Full Training Pipeline Integration ===")
    # This runs the full flow: Train Scouts -> Mine -> Train Experts -> Optimize Threshold -> Generate Submission
    pipeline = TrainingPipeline(config=FastConfig)

    # We run with debug=True to use the small dataset logic inside the pipeline
    pipeline.run(load_cached_data=False, debug=True)

    # Validation of Artifacts
    assert os.path.exists(
        FastConfig.CACHE_BEST_THRESHOLD
    ), "Best threshold file not found."
    assert os.path.exists(FastConfig.SUBMISSION_PATH), "Submission file not found."

    # Validate Submission Content
    df_sub = pd.read_csv(FastConfig.SUBMISSION_PATH)
    assert "contact_id" in df_sub.columns, "Submission missing contact_id."
    assert "contact" in df_sub.columns, "Submission missing contact column."
    assert (
        df_sub["contact"].isin([0, 1]).all()
    ), "Submission contains non-binary predictions."

    print(
        f"Pipeline completed successfully. Submission generated at {FastConfig.SUBMISSION_PATH}"
    )
    print(f"Submission rows: {len(df_sub)}")
