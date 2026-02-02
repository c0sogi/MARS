import os
import shutil
import numpy as np
import pandas as pd
import warnings
import logging

# Import from provided library
from library.config import Config
from library.trainer import Trainer
from library.utils import seed_everything

# Suppress warnings and logs for clean output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("lightgbm").setLevel(logging.ERROR)


def create_mini_dataset(
    original_meta_path,
    original_tracking_path,
    new_meta_path,
    new_tracking_path,
    n_samples=500,
):
    """
    Creates a smaller version of metadata and tracking files for demonstration purposes.
    Ensures tracking data matches the sampled plays in metadata.
    """
    print(f"Creating mini dataset from {os.path.basename(original_meta_path)}...")

    # Load and sample metadata
    df_meta = pd.read_csv(original_meta_path)
    if len(df_meta) > n_samples:
        df_meta = df_meta.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # Save mini metadata
    df_meta.to_csv(new_meta_path, index=False)

    # Identify unique plays in the sampled metadata
    sampled_plays = df_meta["game_play"].unique()

    # Load and filter tracking data
    # Note: Test tracking path might be different, but logic is same.
    # We read the tracking file provided in the arguments.
    if os.path.exists(original_tracking_path):
        print(
            f"Filtering tracking data from {os.path.basename(original_tracking_path)}..."
        )
        # Read in chunks or full? The files are ~1M rows, full read is fine for this environment.
        df_tracking = pd.read_csv(original_tracking_path)

        # Filter for relevant plays
        df_tracking_mini = df_tracking[
            df_tracking["game_play"].isin(sampled_plays)
        ].copy()

        # If no tracking data found for these plays (unlikely if sampled from valid meta),
        # keep a dummy row to prevent schema errors, or just save empty.
        # However, feature extractor expects data.
        if df_tracking_mini.empty:
            print(
                "Warning: No matching tracking data found for sampled plays. Taking random tracking sample."
            )
            df_tracking_mini = df_tracking.head(1000).copy()

        df_tracking_mini.to_csv(new_tracking_path, index=False)
    else:
        # If tracking file doesn't exist (e.g. test tracking in some envs), create dummy
        print(f"Warning: {original_tracking_path} not found. Creating dummy tracking.")
        pd.DataFrame(
            columns=["game_play", "step", "nfl_player_id", "acceleration"]
        ).to_csv(new_tracking_path, index=False)

    return len(df_meta)


def setup_demo_config():
    """
    Overrides the default configuration for a quick demonstration run.
    """
    print("Configuring demo parameters...")

    # 1. Set up paths for mini datasets
    demo_dir = os.path.join("./working", "demo_run")
    os.makedirs(demo_dir, exist_ok=True)

    # Define mini file paths
    mini_train_meta = os.path.join(demo_dir, "mini_train_metadata.csv")
    mini_val_meta = os.path.join(demo_dir, "mini_val_metadata.csv")
    mini_test_meta = os.path.join(demo_dir, "mini_test_metadata.csv")

    mini_train_track = os.path.join(demo_dir, "mini_train_tracking.csv")
    mini_test_track = os.path.join(demo_dir, "mini_test_tracking.csv")

    # 2. Create Mini Datasets
    # Train (Sample 2000 rows)
    create_mini_dataset(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_TRACKING_PATH,
        mini_train_meta,
        mini_train_track,
        n_samples=2000,
    )

    # Val (Sample 500 rows)
    create_mini_dataset(
        Config.VAL_METADATA_PATH,
        Config.TRAIN_TRACKING_PATH,
        mini_val_meta,
        mini_train_track,
        n_samples=500,
    )
    # Note: Val usually uses train tracking file structure in this dataset

    # Test (Sample 500 rows)
    test_len = create_mini_dataset(
        Config.TEST_METADATA_PATH,
        Config.TEST_TRACKING_PATH,
        mini_test_meta,
        mini_test_track,
        n_samples=500,
    )

    # 3. Override Config Class Attributes
    Config.EXPERIMENT_ID = "demo_run"
    Config.WORKING_DIR = demo_dir

    # Update Data Paths
    Config.TRAIN_METADATA_PATH = mini_train_meta
    Config.VAL_METADATA_PATH = mini_val_meta
    Config.TEST_METADATA_PATH = mini_test_meta

    Config.TRAIN_TRACKING_PATH = mini_train_track
    Config.TEST_TRACKING_PATH = mini_test_track

    # Update Cache Paths (to ensure we don't load old full features)
    Config.CACHE_PATHS = {
        "train_features": os.path.join(demo_dir, "features_train_full.parquet"),
        "val_features": os.path.join(demo_dir, "features_val_full.parquet"),
        "test_features": os.path.join(demo_dir, "features_test_full.parquet"),
        "hard_negatives": os.path.join(demo_dir, "hard_negative_indices.npy"),
        "scout_preds": os.path.join(demo_dir, "scout_predictions.npy"),
    }

    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # 4. Reduce Model Complexity for Speed
    # LightGBM
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 31

    # XGBoost
    Config.XGB_PARAMS["n_estimators"] = 10

    # CatBoost
    Config.CATBOOST_PARAMS["iterations"] = 10

    # Training Loop
    Config.TRAINING["EARLY_STOPPING_ROUNDS"] = 5

    # Disable Gating for demo to ensure we get features for all sampled rows
    # (or keep enabled but ensure threshold isn't too aggressive for random samples)
    # We'll keep it enabled to test the logic, but relax window slightly if needed.
    # Default config is fine.

    return test_len


def verify_submission(expected_rows):
    """
    Verifies the generated submission file.
    """
    sub_path = Config.SUBMISSION_PATH
    print(f"\nVerifying submission at {sub_path}...")

    if not os.path.exists(sub_path):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(sub_path)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())

    # Checks
    if len(df_sub) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} rows in submission, found {len(df_sub)}."
        )

    if list(df_sub.columns) != ["contact_id", "contact"]:
        raise AssertionError(f"Invalid columns: {df_sub.columns}")

    if not df_sub["contact"].isin([0, 1]).all():
        raise AssertionError("Predictions must be binary (0 or 1).")

    print("Verification Passed: Submission format is correct.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # 2. Configure for Demo (Create mini datasets and override Config)
    expected_test_rows = setup_demo_config()

    # 3. Instantiate Trainer
    # This will initialize DataManager, FeatureExtractor, etc. using the modified Config
    trainer = Trainer()

    # 4. Run Pipeline
    # This executes: Train Scouts -> Mine Negatives -> Train Experts -> Optimize Threshold -> Predict
    print("\nStarting Pipeline Execution...")
    trainer.run()

    # 5. Verify Results
    verify_submission(expected_test_rows)

    print("\nDemo completed successfully.")
