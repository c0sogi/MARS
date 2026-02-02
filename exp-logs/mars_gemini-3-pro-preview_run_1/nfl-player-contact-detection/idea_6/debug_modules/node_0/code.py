import os
import pandas as pd
import numpy as np
import logging
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.pipeline import IHNMEPipeline
from library.utils import seed_everything, setup_logger

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_mini_datasets():
    """
    Creates small subsets of the original data to allow the pipeline
    to run end-to-end quickly for demonstration purposes.
    """
    print("Creating mini-datasets for demonstration...")

    # Define paths for mini datasets
    mini_train_meta_path = os.path.join(Config.WORKING_DIR, "mini_train_metadata.csv")
    mini_val_meta_path = os.path.join(Config.WORKING_DIR, "mini_val_metadata.csv")
    mini_test_meta_path = os.path.join(Config.WORKING_DIR, "mini_test_metadata.csv")

    mini_train_track_path = os.path.join(Config.WORKING_DIR, "mini_train_tracking.csv")
    mini_test_track_path = os.path.join(Config.WORKING_DIR, "mini_test_tracking.csv")

    # 1. Load original metadata
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Select a few unique plays to keep data consistent
    # We pick 2 plays for training, 1 for validation, 1 for testing
    train_plays = df_train_meta["game_play"].unique()[:2]
    val_plays = df_val_meta["game_play"].unique()[:1]
    test_plays = df_test_meta["game_play"].unique()[:1]

    # Filter metadata
    mini_train_meta = df_train_meta[df_train_meta["game_play"].isin(train_plays)].copy()
    mini_val_meta = df_val_meta[df_val_meta["game_play"].isin(val_plays)].copy()
    mini_test_meta = df_test_meta[df_test_meta["game_play"].isin(test_plays)].copy()

    # Save mini metadata
    mini_train_meta.to_csv(mini_train_meta_path, index=False)
    mini_val_meta.to_csv(mini_val_meta_path, index=False)
    mini_test_meta.to_csv(mini_test_meta_path, index=False)

    print(f"Mini Train Metadata: {len(mini_train_meta)} rows")
    print(f"Mini Val Metadata: {len(mini_val_meta)} rows")
    print(f"Mini Test Metadata: {len(mini_test_meta)} rows")

    # 3. Create corresponding tracking data
    # We need to filter the large tracking files to match the selected plays

    # Train/Val Tracking (Source: Train Tracking File)
    print("Filtering train tracking data...")
    needed_plays_train = np.concatenate([train_plays, val_plays])

    # Read full tracking file (it fits in memory given the specs, but we filter immediately)
    df_train_track = pd.read_csv(Config.TRAIN_TRACKING_PATH)
    mini_train_track = df_train_track[
        df_train_track["game_play"].isin(needed_plays_train)
    ].copy()
    mini_train_track.to_csv(mini_train_track_path, index=False)

    # Test Tracking
    print("Filtering test tracking data...")
    df_test_track = pd.read_csv(Config.TEST_TRACKING_PATH)
    mini_test_track = df_test_track[df_test_track["game_play"].isin(test_plays)].copy()
    mini_test_track.to_csv(mini_test_track_path, index=False)

    print(f"Mini Train Tracking: {len(mini_train_track)} rows")
    print(f"Mini Test Tracking: {len(mini_test_track)} rows")

    return {
        "train_meta": mini_train_meta_path,
        "val_meta": mini_val_meta_path,
        "test_meta": mini_test_meta_path,
        "train_track": mini_train_track_path,
        "test_track": mini_test_track_path,
    }


def configure_demo_environment(paths):
    """
    Overrides the global Config class attributes to use the mini-datasets
    and reduce model complexity for speed.
    """
    print("\nConfiguring environment for demo run...")

    # Override Paths
    Config.TRAIN_METADATA_PATH = paths["train_meta"]
    Config.VAL_METADATA_PATH = paths["val_meta"]
    Config.TEST_METADATA_PATH = paths["test_meta"]
    Config.TRAIN_TRACKING_PATH = paths["train_track"]
    Config.TEST_TRACKING_PATH = paths["test_track"]

    # Set a specific working directory for demo artifacts
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Override Model Hyperparameters for Speed
    # Reduce estimators to a bare minimum to verify the training loop works
    Config.LGBM_SCOUT_PARAMS["n_estimators"] = 10
    Config.LGBM_EXPERT_PARAMS["n_estimators"] = 10
    Config.XGB_EXPERT_PARAMS["n_estimators"] = 10

    # Reduce early stopping rounds
    Config.EARLY_STOPPING_ROUNDS = 2

    # Reduce sampling ratios to keep datasets tiny
    Config.SCOUT_NEG_RATIO = 1
    Config.EXPERT_RANDOM_NEG_RATIO = 1

    # Ensure reproducibility
    seed_everything(Config.SEED)


def run_demo():
    # 1. Setup Data
    # We create mini datasets in the default working dir first
    paths = create_mini_datasets()

    # 2. Configure Config
    configure_demo_environment(paths)

    # 3. Initialize Pipeline
    print("\nInitializing IHNME Pipeline...")
    pipeline = IHNMEPipeline()

    # 4. Run Pipeline
    # This executes: Data Prep -> Scout Train -> Mining -> Expert Train -> Optimization -> Inference
    print("Executing Pipeline...")
    pipeline.run()

    # 5. Verify Outputs
    print("\nVerifying outputs...")
    submission_path = Config.SUBMISSION_PATH

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Check columns
    expected_cols = ["contact_id", "contact"]
    if not all(col in df_sub.columns for col in expected_cols):
        raise ValueError(
            f"Submission missing required columns. Found: {df_sub.columns}"
        )

    # Check values
    if not df_sub["contact"].isin([0, 1]).all():
        raise ValueError("Submission contains non-binary values in 'contact' column.")

    print(
        "Verification Successful: Pipeline ran from end-to-end and produced a valid submission."
    )


if __name__ == "__main__":
    run_demo()
