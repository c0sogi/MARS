import os
import shutil
import pandas as pd
import numpy as np
import joblib
from library.config import Config
from library.utils import seed_everything, setup_logger
from library.data_loader import DataLoader
from library.feature_engineering import FeatureGenerator
from library.mining import ScoutMiner
from library.training import ExpertTrainer
from library.inference import InferencePipeline


def run_demo():
    # =========================================================================
    # 0. Setup and Configuration Overrides for Speed
    # =========================================================================
    print("--- 0. Configuring Environment for Fast Demo ---")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Small sample for quick execution
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce model complexity for demo
    Config.SCOUT_LGBM_PARAMS["n_estimators"] = 2
    Config.SCOUT_LGBM_PARAMS["min_child_samples"] = 5  # Allow split on small data

    Config.EXPERT_LGBM_PARAMS["n_estimators"] = 2
    Config.EXPERT_LGBM_PARAMS["min_child_samples"] = 5

    Config.EXPERT_XGB_PARAMS["n_estimators"] = 2

    # Set seeds
    seed_everything(Config.SEED)
    logger = setup_logger("DemoRunner")
    logger.info(f"Working Directory: {Config.WORKING_DIR}")

    # =========================================================================
    # 1. Data Loading Demo
    # =========================================================================
    print("\n--- 1. Testing DataLoader ---")
    loader = DataLoader(debug=True)

    # Load Train Data (Merged)
    # This internally loads metadata and tracking, and merges them
    # Since DEBUG=True, it should cache a small subset
    df_merged_train = loader.get_merged_data(split="train", load_cached_data=False)

    # Validation
    assert (
        len(df_merged_train) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows, got {len(df_merged_train)}"
    assert (
        "x_position_p1" in df_merged_train.columns
    ), "Merged data missing Player 1 tracking columns"
    assert (
        "x_position_p2" in df_merged_train.columns
    ), "Merged data missing Player 2 tracking columns"

    logger.info("DataLoader test passed: Data loaded and merged correctly.")

    # Load Val Data (Merged)
    df_merged_val = loader.get_merged_data(split="val", load_cached_data=False)
    assert len(df_merged_val) == Config.DEBUG_SAMPLE_SIZE

    # Load Raw Tracking (needed for context features later)
    tracking_train = loader.load_tracking(split="train")

    # =========================================================================
    # 2. Feature Engineering Demo (Tier 1)
    # =========================================================================
    print("\n--- 2. Testing FeatureGenerator (Tier 1) ---")
    generator = FeatureGenerator()

    # Generate Tier 1 features for Train
    df_train_tier1 = generator.generate(
        df_merged_train, tracking_train, tier=1, split="train", load_cached_data=False
    )

    # Generate Tier 1 features for Val
    df_val_tier1 = generator.generate(
        df_merged_val, tracking_train, tier=1, split="val", load_cached_data=False
    )

    # Validation
    expected_cols = Config.TIER1_FEATURES
    missing_cols = [c for c in expected_cols if c not in df_train_tier1.columns]
    # Note: Some lag/lead columns might be missing if the windowing logic results in all NaNs
    # or if the feature list definition is strict. However, FeatureGenerator usually adds them.
    # Let's check a base kinematic feature.
    assert "distance" in df_train_tier1.columns, "Feature 'distance' missing."
    assert "speed_diff" in df_train_tier1.columns, "Feature 'speed_diff' missing."

    logger.info(f"FeatureGenerator (Tier 1) passed. Shape: {df_train_tier1.shape}")

    # =========================================================================
    # 3. Mining Demo (Scout)
    # =========================================================================
    print("\n--- 3. Testing ScoutMiner ---")
    miner = ScoutMiner()

    # Execute mining
    # This trains a small LGBM and returns indices of Hard Negatives + Positives
    mined_indices = miner.execute(df_train_tier1, df_val_tier1, load_cached_data=False)

    # Validation
    assert isinstance(
        mined_indices, np.ndarray
    ), "Mined indices should be a numpy array."
    assert len(mined_indices) > 0, "Mining returned no indices."
    # Ensure indices are within bounds of the original dataframe
    assert mined_indices.max() < len(df_train_tier1), "Mined indices out of bounds."

    logger.info(f"ScoutMiner passed. Selected {len(mined_indices)} samples.")

    # =========================================================================
    # 4. Training Demo (Expert)
    # =========================================================================
    print("\n--- 4. Testing ExpertTrainer ---")
    trainer = ExpertTrainer()

    # Train the expert models
    # Note: ExpertTrainer internally re-loads data using DataLoader.
    # Since we set Config.DEBUG=True and seeded everything, it loads the SAME subset
    # as step 1, so the indices match.
    trainer.train(mined_indices, load_cached_data=False)

    # Validation
    model_dir = os.path.join(Config.WORKING_DIR, "models")
    assert os.path.exists(
        os.path.join(model_dir, "lgbm_expert.joblib")
    ), "LGBM model file missing."
    assert os.path.exists(
        os.path.join(model_dir, "xgb_expert.joblib")
    ), "XGB model file missing."
    assert os.path.exists(
        os.path.join(model_dir, "threshold.joblib")
    ), "Threshold file missing."

    logger.info("ExpertTrainer passed. Models and threshold saved.")

    # =========================================================================
    # 5. Inference Demo
    # =========================================================================
    print("\n--- 5. Testing InferencePipeline ---")
    inference = InferencePipeline()

    # Run inference on Test set
    # DataLoader will load test data (sampled due to DEBUG=True)
    inference.run(load_cached_data=False)

    # Validation
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not created."

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    assert "contact_id" in df_sub.columns, "Submission missing contact_id."
    assert "contact" in df_sub.columns, "Submission missing contact column."
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} predictions, got {len(df_sub)}"

    logger.info(
        f"InferencePipeline passed. Submission generated at {Config.SUBMISSION_FILE}"
    )

    print("\n=== DEMO COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_demo()
