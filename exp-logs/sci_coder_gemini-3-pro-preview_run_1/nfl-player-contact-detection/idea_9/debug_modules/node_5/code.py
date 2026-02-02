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

    # 1. Load Tracking Data First (to determine available plays)
    # We read a chunk of tracking data to define the universe of playable games for the demo
    tracking_chunk = pd.read_csv(config.TRAIN_TRACKING_PATH, nrows=50000)
    available_plays = set(tracking_chunk["game_play"].unique())

    # 2. Scan Metadata for Matching Plays AND Positive Contacts
    # We must ensure the mini-dataset contains at least some contact events (contact=1)
    # otherwise the training logic (balancing) will produce an empty dataset.
    iter_csv = pd.read_csv(config.TRAIN_METADATA_PATH, iterator=True, chunksize=10000)

    candidate_meta = []
    found_contacts = 0

    for chunk in iter_csv:
        # Filter for plays present in our tracking chunk
        filtered = chunk[chunk["game_play"].isin(available_plays)]
        if not filtered.empty:
            candidate_meta.append(filtered)
            found_contacts += (filtered["contact"] == 1).sum()

            # Stop if we have enough data and enough contacts
            if found_contacts > 10 and len(pd.concat(candidate_meta)) > 500:
                break

    if candidate_meta:
        mini_train_meta = pd.concat(candidate_meta)
    else:
        # Fallback (should not happen given dataset size)
        mini_train_meta = pd.read_csv(config.TRAIN_METADATA_PATH, nrows=200)

    # Prioritize keeping rows from plays that have contacts
    plays_with_contacts = mini_train_meta[mini_train_meta["contact"] == 1][
        "game_play"
    ].unique()

    if len(plays_with_contacts) > 0:
        # Keep all rows for plays that have contacts, plus some others if needed
        mini_train_meta = mini_train_meta[
            mini_train_meta["game_play"].isin(plays_with_contacts)
        ]

    # Cap size to keep it "mini" but ensure we don't lose the positives
    if len(mini_train_meta) > 2000:
        pos_rows = mini_train_meta[mini_train_meta["contact"] == 1]
        neg_rows = mini_train_meta[mini_train_meta["contact"] == 0].head(
            2000 - len(pos_rows)
        )
        mini_train_meta = pd.concat([pos_rows, neg_rows])

    # Save aligned mini datasets
    mini_train_meta_path = os.path.join(base_dir, "mini_train_metadata.csv")
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
