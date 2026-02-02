import os
import shutil
import pandas as pd
import numpy as np
import joblib

# Import from provided library
from library.config import Config
from library.utils import seed_everything
from library.data_manager import DataManager
from library.feature_engine import FeatureEngine
from library.trainer import Trainer
from library.inference import InferencePipeline


def run_demo():
    # -------------------------------------------------------------------------
    # 0. Setup and Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    print(">>> Setting up demonstration environment...")

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Override Hyperparameters for fast execution
    print(">>> Overriding hyperparameters for speed...")
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8

    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["max_depth"] = 3

    Config.HGB_PARAMS["max_iter"] = 10
    Config.HGB_PARAMS["max_leaf_nodes"] = 8

    # Define a demo working directory to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(
        Config.SUBMISSION_DIR, "mini_sample_submission.csv"
    )

    # Clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Data Manager Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> [1] Testing DataManager...")
    dm = DataManager()

    # Load a small sample of training data
    # We use a sample size sufficient to likely include both positive and negative examples
    sample_size = 5000
    df_train_raw = dm.load_train_data(load_cached_data=False, sample_size=sample_size)

    print(f"Loaded raw train sample shape: {df_train_raw.shape}")

    # Assertions
    assert not df_train_raw.empty, "Training data sample is empty."
    assert "distance" in df_train_raw.columns, "Merged data missing 'distance' column."
    assert (
        "x_position_p1" in df_train_raw.columns
    ), "Merged data missing player 1 tracking."

    # Check Sentinel Value for Ground
    ground_rows = df_train_raw[df_train_raw["nfl_player_id_2"] == "G"]
    if not ground_rows.empty:
        # Check if distance is -1.0 (Sentinel)
        assert (
            ground_rows["distance"] == Config.GROUND_DISTANCE_SENTINEL
        ).all(), "Sentinel value strategy for Ground interactions failed."

    print("DataManager tests passed.")

    # -------------------------------------------------------------------------
    # 2. Feature Engine Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> [2] Testing FeatureEngine...")
    fe = FeatureEngine()

    # Ensure the cache manager inside FE uses our overridden directory
    fe.cache_manager.cache_dir = Config.WORKING_DIR
    fe.data_manager.cache_manager.cache_dir = Config.WORKING_DIR

    # Process the sample
    df_features = fe.process_train(load_cached_data=False, sample_size=sample_size)

    print(f"Processed features shape: {df_features.shape}")

    # Assertions
    expected_cols = ["proj_u_p1", "proj_v_p1", "acc_u_p1", "jerk_u_p1"]
    for col in expected_cols:
        assert col in df_features.columns, f"Missing feature column: {col}"

    # Check Lag Generation
    lag_col = "distance_lag_1"
    assert lag_col in df_features.columns, "Lag features not generated."

    # Check Gating
    assert isinstance(df_features, pd.DataFrame)

    print("FeatureEngine tests passed.")

    # -------------------------------------------------------------------------
    # 3. Trainer Demonstration (Full Pipeline)
    # -------------------------------------------------------------------------
    print("\n>>> [3] Testing Trainer (Full Pipeline)...")
    trainer = Trainer()

    # Update internal paths for Trainer to use demo directory
    trainer.models_dir = os.path.join(Config.WORKING_DIR, "models")
    trainer.scout_dir = os.path.join(trainer.models_dir, "scouts")
    trainer.expert_dir = os.path.join(trainer.models_dir, "experts")
    trainer.cache_manager.cache_dir = Config.WORKING_DIR
    trainer.feature_engine.cache_manager.cache_dir = Config.WORKING_DIR

    os.makedirs(trainer.scout_dir, exist_ok=True)
    os.makedirs(trainer.expert_dir, exist_ok=True)

    # Run the trainer with the small sample size
    # This executes: Train Scouts -> Mine Hard Negs -> Train Experts -> Eval -> Predict Test
    trainer.run(load_cached_data=False, sample_size=sample_size)

    # Verify Artifacts
    print("Verifying Trainer artifacts...")

    # 1. Models
    for model_type in ["lgbm", "xgb", "hgb"]:
        scout_path = os.path.join(trainer.scout_dir, f"{model_type}_model.joblib")
        expert_path = os.path.join(trainer.expert_dir, f"{model_type}_model.joblib")
        assert os.path.exists(scout_path), f"Scout model {model_type} not saved."
        assert os.path.exists(expert_path), f"Expert model {model_type} not saved."

    # 2. Threshold
    thresh_path = os.path.join(trainer.models_dir, "best_threshold.npy")
    assert os.path.exists(thresh_path), "Best threshold file not saved."

    # 3. Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    # Verify Submission Content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "contact_id" in sub_df.columns
    assert "contact" in sub_df.columns
    assert (
        sub_df["contact"].isin([0, 1]).all()
    ), "Submission contains non-binary values."

    print("Trainer pipeline execution successful.")

    # -------------------------------------------------------------------------
    # 4. Inference Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> [4] Testing InferencePipeline (Standalone)...")

    # Instantiate InferencePipeline
    inference = InferencePipeline()

    # Update paths to point to the artifacts created by the Trainer
    inference.models_dir = trainer.models_dir
    inference.expert_dir = trainer.expert_dir
    inference.threshold_path = thresh_path
    inference.feature_engine.cache_manager.cache_dir = Config.WORKING_DIR

    # Delete previous submission to verify regeneration
    if os.path.exists(Config.SUBMISSION_PATH):
        os.remove(Config.SUBMISSION_PATH)

    # Run Inference
    inference.generate_submission(load_cached_data=True, sample_size=sample_size)

    # Verify
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), "InferencePipeline failed to generate submission."

    print("InferencePipeline tests passed.")
    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
