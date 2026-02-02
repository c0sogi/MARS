import os
import pandas as pd
import numpy as np
import shutil
import logging
from library.config import Config
from library.utils import seed_everything, setup_logging
from library.pipeline import TrainingPipeline


def create_mini_dataset(
    metadata_path, tracking_path, out_meta_path, out_track_path, n_samples=2000
):
    """
    Creates a smaller version of the dataset for demonstration purposes.
    Ensures tracking data matches the sampled plays in metadata.
    """
    print(f"Creating mini dataset from {metadata_path}...")

    # Load full metadata
    df_meta = pd.read_csv(metadata_path)

    # Sample metadata
    if len(df_meta) > n_samples:
        df_meta_mini = df_meta.sample(n=n_samples, random_state=42).reset_index(
            drop=True
        )
    else:
        df_meta_mini = df_meta.copy()

    # Get unique plays in the sample
    unique_plays = df_meta_mini["game_play"].unique()

    # Load full tracking data
    # Note: In a real scenario with massive files, we might iterate/chunk.
    # Here, files are manageable (~100MB).
    df_track = pd.read_csv(tracking_path)

    # Filter tracking data for relevant plays
    df_track_mini = df_track[df_track["game_play"].isin(unique_plays)].copy()

    # Save mini datasets
    df_meta_mini.to_csv(out_meta_path, index=False)
    df_track_mini.to_csv(out_track_path, index=False)

    print(f"Saved mini metadata ({len(df_meta_mini)} rows) to {out_meta_path}")
    print(f"Saved mini tracking ({len(df_track_mini)} rows) to {out_track_path}")

    return len(df_meta_mini)


def run_demo():
    # 1. Setup
    seed_everything(42)
    setup_logging("demo_execution/demo.log")

    # Define working paths
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    mini_train_meta = os.path.join(demo_dir, "mini_train_metadata.csv")
    mini_val_meta = os.path.join(demo_dir, "mini_val_metadata.csv")
    mini_test_meta = os.path.join(demo_dir, "mini_test_metadata.csv")

    mini_train_track = os.path.join(demo_dir, "mini_train_tracking.csv")
    mini_test_track = os.path.join(demo_dir, "mini_test_tracking.csv")

    # 2. Create Mini Datasets
    # Temporary paths for disjoint tracking data
    mini_train_track_part = os.path.join(demo_dir, "mini_train_track_part.csv")
    mini_val_track_part = os.path.join(demo_dir, "mini_val_track_part.csv")

    # Train
    create_mini_dataset(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_TRACKING_PATH,
        mini_train_meta,
        mini_train_track_part,
        n_samples=2000,
    )

    # Val (Uses train tracking file)
    create_mini_dataset(
        Config.VAL_METADATA_PATH,
        Config.TRAIN_TRACKING_PATH,
        mini_val_meta,
        mini_val_track_part,
        n_samples=500,
    )

    # Merge Tracking Data for the 'Train' split (which includes Val in the physical file)
    print("Merging mini tracking datasets...")
    df_track_train = pd.read_csv(mini_train_track_part)
    df_track_val = pd.read_csv(mini_val_track_part)
    # Concatenate and save to the path expected by Config
    pd.concat([df_track_train, df_track_val], ignore_index=True).to_csv(
        mini_train_track, index=False
    )

    # Clean up temporary files
    if os.path.exists(mini_train_track_part):
        os.remove(mini_train_track_part)
    if os.path.exists(mini_val_track_part):
        os.remove(mini_val_track_part)

    # Test
    n_test_samples = create_mini_dataset(
        Config.TEST_METADATA_PATH,
        Config.TEST_TRACKING_PATH,
        mini_test_meta,
        mini_test_track,
        n_samples=500,
    )

    # 3. Runtime Configuration Overrides
    # We modify the Config class attributes directly to point to our mini datasets
    # and speed up training.
    print("\nOverriding Config for Demo...")

    # Paths
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_METADATA_PATH = mini_train_meta
    Config.VAL_METADATA_PATH = mini_val_meta
    Config.TEST_METADATA_PATH = mini_test_meta
    Config.TRAIN_TRACKING_PATH = mini_train_track
    Config.TEST_TRACKING_PATH = mini_test_track

    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    Config.CACHE_TRAIN_FEATURES = os.path.join(demo_dir, "features_train.parquet")
    Config.CACHE_VAL_FEATURES = os.path.join(demo_dir, "features_val.parquet")
    Config.CACHE_TEST_FEATURES = os.path.join(demo_dir, "features_test.parquet")
    Config.CACHE_HARD_NEGATIVES = os.path.join(demo_dir, "hard_negative_indices.npy")
    Config.MODEL_DIR = os.path.join(demo_dir, "models")
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    # Hyperparameters (Speed Optimization)
    # Reduce estimators to minimal values for fast execution
    Config.LGBM_PARAMS["n_estimators"] = 20
    Config.XGB_PARAMS["n_estimators"] = 20
    Config.HGB_PARAMS["max_iter"] = 20

    # 4. Execute Pipeline
    print("\nInitializing Pipeline...")
    pipeline = TrainingPipeline()

    # Phase 1: Train Scouts
    # We set load_cached_data=False to ensure we process our new mini datasets
    pipeline.train_scouts(load_cached_data=False)

    # Phase 2: Mine Hard Negatives
    pipeline.mine_hard_negatives(load_cached_data=False)

    # Phase 3: Train Expert
    pipeline.train_expert(load_cached_data=False)

    # Phase 4: Evaluation
    best_mcc = pipeline.evaluate(load_cached_data=False)
    print(f"\nPipeline Evaluation MCC: {best_mcc}")

    # Phase 5: Inference
    pipeline.inference(load_cached_data=False)

    # 5. Validation
    print("\nValidating Output...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")

    # Check rows match test input
    if len(df_sub) != n_test_samples:
        raise AssertionError(
            f"Submission row count {len(df_sub)} does not match test input {n_test_samples}"
        )

    # Check columns
    expected_cols = ["contact_id", "contact"]
    if not all(col in df_sub.columns for col in expected_cols):
        raise AssertionError(f"Submission columns missing. Found: {df_sub.columns}")

    print("Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
