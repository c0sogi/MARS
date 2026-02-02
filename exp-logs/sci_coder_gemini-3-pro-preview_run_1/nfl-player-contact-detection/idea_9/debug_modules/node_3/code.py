import os
import pandas as pd
import numpy as np
import shutil
import sys

# Import library modules
import library.config as config
import library.model_zoo as model_zoo
from library.trainer import MiningTrainer
from library.inference import InferenceManager
from library.utils import seed_everything


def create_mini_datasets(base_dir):
    """
    Creates small subsets of the original data to verify the pipeline runs correctly
    and quickly.
    """
    print("Creating mini datasets for rapid demonstration...")

    # 1. Load a small sample of Train Metadata
    # We read enough rows to get a few unique game_plays
    full_train_meta = pd.read_csv(config.TRAIN_METADATA_PATH, nrows=1000)
    unique_plays = full_train_meta["game_play"].unique()[:2]  # Take 2 plays
    mini_train_meta = full_train_meta[
        full_train_meta["game_play"].isin(unique_plays)
    ].copy()

    # Save mini train metadata
    mini_train_meta_path = os.path.join(base_dir, "mini_train_metadata.csv")
    mini_train_meta.to_csv(mini_train_meta_path, index=False)

    # 2. Create corresponding Tracking Data
    # We need tracking data that matches the game_plays in our metadata
    # Since tracking data is huge, we iterate or read chunks, but for demo we read head
    # and hope for overlap, or better: we filter the full tracking file if possible.
    # Given read-only input, we'll read a chunk large enough to likely cover the first few plays
    # or just filter based on the IDs we selected if they appear early.
    # Note: The provided metadata generation script shuffles plays.
    # Let's rely on reading the first chunk of tracking and filtering metadata to match THAT.

    # Read first 50k rows of tracking
    tracking_chunk = pd.read_csv(config.TRAIN_TRACKING_PATH, nrows=50000)
    available_plays = tracking_chunk["game_play"].unique()

    # Filter metadata to match available tracking plays
    mini_train_meta = full_train_meta[
        full_train_meta["game_play"].isin(available_plays)
    ].head(500)
    if mini_train_meta.empty:
        # Fallback: Just use the tracking chunk as is and make metadata match it artificially if needed
        # But usually first rows of metadata and tracking correspond to same games in raw data,
        # though metadata script shuffled.
        # Let's pick plays from tracking chunk and find them in full metadata
        target_plays = set(available_plays)
        # Read metadata in chunks to find matching plays
        iter_csv = pd.read_csv(
            config.TRAIN_METADATA_PATH, iterator=True, chunksize=10000
        )
        mini_train_meta = pd.DataFrame()
        for chunk in iter_csv:
            filtered = chunk[chunk["game_play"].isin(target_plays)]
            if not filtered.empty:
                mini_train_meta = pd.concat([mini_train_meta, filtered])
                if len(mini_train_meta) > 200:
                    break
        mini_train_meta = mini_train_meta.head(200)

    # Save aligned mini datasets
    mini_train_meta.to_csv(mini_train_meta_path, index=False)

    # Filter tracking to match the final metadata selection
    final_plays = mini_train_meta["game_play"].unique()
    mini_tracking = tracking_chunk[tracking_chunk["game_play"].isin(final_plays)].copy()
    mini_tracking_path = os.path.join(base_dir, "mini_train_tracking.csv")
    mini_tracking.to_csv(mini_tracking_path, index=False)

    # 3. Create Val Metadata (Use a subset of train for demo purposes)
    mini_val_meta_path = os.path.join(base_dir, "mini_val_metadata.csv")
    mini_train_meta.head(50).to_csv(mini_val_meta_path, index=False)

    # 4. Create Test Metadata/Tracking
    # We use the sample submission to guide test metadata
    test_meta = pd.read_csv(config.TEST_METADATA_PATH, nrows=100)
    mini_test_meta_path = os.path.join(base_dir, "mini_test_metadata.csv")
    test_meta.to_csv(mini_test_meta_path, index=False)

    # For test tracking, we just use the train tracking structure for the demo
    # (since we can't easily match test IDs without reading full test tracking)
    # or read head of test tracking.
    test_tracking_chunk = pd.read_csv(config.TEST_TRACKING_PATH, nrows=5000)
    mini_test_tracking_path = os.path.join(base_dir, "mini_test_tracking.csv")
    test_tracking_chunk.to_csv(mini_test_tracking_path, index=False)

    print(f"Mini datasets created in {base_dir}")
    return {
        "train_meta": mini_train_meta_path,
        "train_track": mini_tracking_path,
        "val_meta": mini_val_meta_path,
        "test_meta": mini_test_meta_path,
        "test_track": mini_test_tracking_path,
    }


def configure_demo_environment(paths, working_dir):
    """
    Monkey-patches the configuration to use mini datasets and fast model settings.
    """
    print("Configuring environment for demo...")

    # Override Paths
    config.TRAIN_METADATA_PATH = paths["train_meta"]
    config.TRAIN_TRACKING_PATH = paths["train_track"]
    config.VAL_METADATA_PATH = paths["val_meta"]
    config.TEST_METADATA_PATH = paths["test_meta"]
    config.TEST_TRACKING_PATH = paths["test_track"]

    # Override Working Directory to isolate demo outputs
    config.WORKING_DIR = working_dir

    # Override Model Hyperparameters for Speed
    # We set N_ESTIMATORS to a very low number to ensure training finishes in seconds
    config.N_ESTIMATORS = 2
    model_zoo.N_ESTIMATORS = 2

    config.EARLY_STOPPING_ROUNDS = 1
    model_zoo.EARLY_STOPPING_ROUNDS = 1

    # Ensure directories exist
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(os.path.join(working_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(working_dir, "data_cache"), exist_ok=True)
    os.makedirs(os.path.join(working_dir, "mining_cache"), exist_ok=True)


if __name__ == "__main__":
    # 1. Setup Demo Directory
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    # 2. Create Data and Configure
    dataset_paths = create_mini_datasets(demo_dir)
    configure_demo_environment(dataset_paths, demo_dir)

    # Set seed for reproducibility
    seed_everything(42)

    # 3. Execute Training Pipeline
    print("\n--- Starting Training Pipeline ---")
    trainer = MiningTrainer()

    # Verify DataFactory loaded the correct mini paths
    print(
        f"Trainer DataFactory Metadata Path: {trainer.train_data_factory.metadata_path}"
    )
    assert (
        trainer.train_data_factory.metadata_path == dataset_paths["train_meta"]
    ), "Trainer did not pick up the patched metadata path."

    # Run the full training curriculum
    # This includes: Scout Model -> Hard Negative Mining -> Expert Ensemble -> Threshold Opt
    ensemble, threshold = trainer.run()

    print("\n--- Training Complete ---")
    print(f"Optimized Threshold: {threshold}")

    # Verify Model Artifacts
    model_dir = os.path.join(demo_dir, "models")
    assert os.path.exists(
        os.path.join(model_dir, "scout_lgbm.joblib")
    ), "Scout model not saved."
    assert os.path.exists(
        os.path.join(model_dir, "expert_lgbm.joblib")
    ), "Expert LGBM not saved."
    assert os.path.exists(
        os.path.join(model_dir, "expert_xgb.joblib")
    ), "Expert XGB not saved."

    # Verify Threshold Cache
    mining_cache_dir = os.path.join(demo_dir, "mining_cache")
    assert os.path.exists(
        os.path.join(mining_cache_dir, "best_threshold.npy")
    ), "Threshold file not saved."

    # 4. Execute Inference Pipeline
    print("\n--- Starting Inference Pipeline ---")

    # Update submission path for the demo
    config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    inference_manager = InferenceManager()
    inference_manager.generate_submission()

    print("\n--- Inference Complete ---")

    # 5. Final Validation
    submission_path = config.SUBMISSION_PATH
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission generated at {submission_path}")
        print(f"Shape: {df_sub.shape}")
        print(df_sub.head())

        # Check columns
        assert "contact_id" in df_sub.columns, "Submission missing contact_id"
        assert "contact" in df_sub.columns, "Submission missing contact"

        # Check values
        unique_vals = df_sub["contact"].unique()
        assert all(
            v in [0, 1] for v in unique_vals
        ), f"Invalid values in contact column: {unique_vals}"

        print("Validation Successful: Pipeline executed end-to-end.")
    else:
        raise FileNotFoundError("Submission file was not generated.")
