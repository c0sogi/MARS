import pandas as pd
import numpy as np
import os
import sys
import shutil
import gc
from library.config import Config
from library.utils import seed_everything, setup_logger
from library.data_loader import DataLoader
from library.ensemble import EnsemblePipeline

# Define paths for the mini-dataset
WORK_DIR = "./working/demo_run"
MINI_DATA_DIR = os.path.join(WORK_DIR, "input")
os.makedirs(MINI_DATA_DIR, exist_ok=True)


def create_mini_dataset():
    """
    Creates a small subset of the data to allow the pipeline to run quickly for demonstration.
    """
    print("Creating mini-dataset for rapid demonstration...")

    # 1. Load original Metadata
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # 2. Sample Plays (2 Train, 1 Val, 1 Test)
    # We select plays that definitely exist to ensure data consistency
    train_plays = train_meta["game_play"].unique()[:2]
    val_plays = val_meta["game_play"].unique()[:1]
    test_plays = test_meta["game_play"].unique()[:1]

    print(f"Selected Train Plays: {train_plays}")
    print(f"Selected Val Plays: {val_plays}")
    print(f"Selected Test Plays: {test_plays}")

    # 3. Filter Metadata
    mini_train_meta = train_meta[train_meta["game_play"].isin(train_plays)].copy()
    mini_val_meta = val_meta[val_meta["game_play"].isin(val_plays)].copy()
    mini_test_meta = test_meta[test_meta["game_play"].isin(test_plays)].copy()

    # Save Mini Metadata
    mini_train_meta_path = os.path.join(MINI_DATA_DIR, "train_meta.csv")
    mini_val_meta_path = os.path.join(MINI_DATA_DIR, "val_meta.csv")
    mini_test_meta_path = os.path.join(MINI_DATA_DIR, "test_meta.csv")

    mini_train_meta.to_csv(mini_train_meta_path, index=False)
    mini_val_meta.to_csv(mini_val_meta_path, index=False)
    mini_test_meta.to_csv(mini_test_meta_path, index=False)

    # 4. Filter and Save Tracking Data
    # We load the full files once, filter, and save small versions
    print("Filtering Tracking Data...")

    # The 'train' source file must contain both training AND validation plays,
    # because the pipeline loads the training file to process the validation split.
    train_source_plays = np.concatenate([train_plays, val_plays])

    # Train Tracking
    train_tracking = pd.read_csv(Config.TRAIN_TRACKING_PATH)
    mini_train_tracking = train_tracking[
        train_tracking["game_play"].isin(train_source_plays)
    ].copy()
    mini_train_track_path = os.path.join(MINI_DATA_DIR, "train_tracking.csv")
    mini_train_tracking.to_csv(mini_train_track_path, index=False)
    del train_tracking

    # Test Tracking
    test_tracking = pd.read_csv(Config.TEST_TRACKING_PATH)
    mini_test_tracking = test_tracking[
        test_tracking["game_play"].isin(test_plays)
    ].copy()
    mini_test_track_path = os.path.join(MINI_DATA_DIR, "test_tracking.csv")
    mini_test_tracking.to_csv(mini_test_track_path, index=False)
    del test_tracking

    # 5. Filter and Save Helmet Data
    print("Filtering Helmet Data...")
    # Train Helmets
    train_helmets = pd.read_csv(Config.TRAIN_HELMETS_PATH)
    mini_train_helmets = train_helmets[
        train_helmets["game_play"].isin(train_source_plays)
    ].copy()
    mini_train_helm_path = os.path.join(MINI_DATA_DIR, "train_helmets.csv")
    mini_train_helmets.to_csv(mini_train_helm_path, index=False)
    del train_helmets

    # Test Helmets
    test_helmets = pd.read_csv(Config.TEST_HELMETS_PATH)
    mini_test_helmets = test_helmets[test_helmets["game_play"].isin(test_plays)].copy()
    mini_test_helm_path = os.path.join(MINI_DATA_DIR, "test_helmets.csv")
    mini_test_helmets.to_csv(mini_test_helm_path, index=False)
    del test_helmets

    gc.collect()

    return {
        "train_meta": mini_train_meta_path,
        "val_meta": mini_val_meta_path,
        "test_meta": mini_test_meta_path,
        "train_track": mini_train_track_path,
        "test_track": mini_test_track_path,
        "train_helm": mini_train_helm_path,
        "test_helm": mini_test_helm_path,
    }


def patch_config(paths):
    """
    Patches the Config class to use the mini-dataset and faster training parameters.
    """
    print("Patching Config for Demo...")

    # Update Paths
    Config.TRAIN_META_PATH = paths["train_meta"]
    Config.VAL_META_PATH = paths["val_meta"]
    Config.TEST_META_PATH = paths["test_meta"]

    Config.TRAIN_TRACKING_PATH = paths["train_track"]
    Config.TEST_TRACKING_PATH = paths["test_track"]

    Config.TRAIN_HELMETS_PATH = paths["train_helm"]
    Config.TEST_HELMETS_PATH = paths["test_helm"]

    Config.WORKING_DIR = WORK_DIR
    Config.SUBMISSION_DIR = WORK_DIR

    # Update Model Parameters for Speed
    # Reduce estimators and depth for quick execution
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["max_depth"] = 3
    Config.XGB_PARAMS["early_stopping_rounds"] = 2

    # Reduce Blending Search
    Config.BLENDING_TRIALS = 5

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)


def run_pipeline_demo():
    """
    Executes the full Ensemble Pipeline using the patched configuration.
    """
    # Initialize Pipeline
    # Note: Config is patched before this instantiation, so the pipeline picks up the changes.
    pipeline = EnsemblePipeline()

    # 1. Training Phase
    print("\n=== Starting Training Phase ===")
    # load_cached_data=False ensures we process our new mini-dataset from scratch
    pipeline.train(load_cached_data=False)

    # Logic Verification: Check if models are trained
    if pipeline.model_tracking.model_pp is None:
        raise AssertionError("Tracking Model (PP) failed to train.")
    if pipeline.model_helmets.model_pp is None:
        raise AssertionError("Helmet Model (PP) failed to train.")

    print("Training completed successfully.")

    # 2. Inference Phase
    print("\n=== Starting Inference Phase ===")
    pipeline.inference(load_cached_data=False)

    # Logic Verification: Check Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not created at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Verify Columns
    expected_cols = ["contact_id", "contact"]
    if not all(col in df_sub.columns for col in expected_cols):
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {df_sub.columns.tolist()}"
        )

    # Verify Values (Binary)
    unique_vals = df_sub["contact"].unique()
    if not all(v in [0, 1] for v in unique_vals):
        raise AssertionError(f"Submission contains non-binary values: {unique_vals}")

    print("Inference and Validation completed successfully.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)
    setup_logger()

    # 2. Create Mini Dataset
    paths = create_mini_dataset()

    # 3. Patch Configuration
    patch_config(paths)

    # 4. Run Pipeline
    run_pipeline_demo()

    print("\nDemo execution finished successfully.")
