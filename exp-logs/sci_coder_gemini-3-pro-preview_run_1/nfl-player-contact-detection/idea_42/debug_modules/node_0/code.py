import os
import pandas as pd
import numpy as np
import shutil

# Import library modules
import library.config as config
import library.data_loader as dl
import library.model_factory as mf
import library.utils as utils
import library.features as features
import library.training_pipeline as tp
import library.inference_pipeline as ip


def create_mini_datasets(demo_dir):
    """
    Creates small subsets of metadata and tracking data for demonstration.
    """
    print("Creating mini-datasets for demonstration...")

    # Define paths
    mini_train_meta_path = os.path.join(demo_dir, "mini_train_metadata.csv")
    mini_val_meta_path = os.path.join(demo_dir, "mini_val_metadata.csv")
    mini_test_meta_path = os.path.join(demo_dir, "mini_test_metadata.csv")
    mini_train_track_path = os.path.join(demo_dir, "mini_train_tracking.csv")
    mini_test_track_path = os.path.join(demo_dir, "mini_test_tracking.csv")

    # 1. Train Metadata (Sample 2 plays)
    df_train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    train_plays = df_train_meta["game_play"].unique()[:2]
    df_mini_train = df_train_meta[df_train_meta["game_play"].isin(train_plays)].copy()
    df_mini_train.to_csv(mini_train_meta_path, index=False)

    # 2. Val Metadata (Sample 1 play)
    df_val_meta = pd.read_csv(config.VAL_METADATA_PATH)
    val_plays = df_val_meta["game_play"].unique()[:1]
    df_mini_val = df_val_meta[df_val_meta["game_play"].isin(val_plays)].copy()
    df_mini_val.to_csv(mini_val_meta_path, index=False)

    # 3. Test Metadata (Sample 1 play)
    df_test_meta = pd.read_csv(config.TEST_METADATA_PATH)
    test_plays = df_test_meta["game_play"].unique()[:1]
    df_mini_test = df_test_meta[df_test_meta["game_play"].isin(test_plays)].copy()
    df_mini_test.to_csv(mini_test_meta_path, index=False)

    # 4. Train Tracking (Filter for Train + Val plays)
    needed_plays = np.concatenate([train_plays, val_plays])
    df_track = pd.read_csv(config.TRAIN_TRACKING_PATH)
    df_mini_track = df_track[df_track["game_play"].isin(needed_plays)].copy()
    df_mini_track.to_csv(mini_train_track_path, index=False)

    # 5. Test Tracking (Filter for Test plays)
    df_test_track = pd.read_csv(config.TEST_TRACKING_PATH)
    df_mini_test_track = df_test_track[
        df_test_track["game_play"].isin(test_plays)
    ].copy()
    df_mini_test_track.to_csv(mini_test_track_path, index=False)

    return {
        "train_meta": mini_train_meta_path,
        "val_meta": mini_val_meta_path,
        "test_meta": mini_test_meta_path,
        "train_track": mini_train_track_path,
        "test_track": mini_test_track_path,
    }


def patch_modules(paths, demo_dir):
    """
    Patches imported modules to use mini-datasets and fast hyperparameters.
    """
    print("Patching modules with demo configuration...")

    # 1. Patch Working Directory
    # We need to update this in all modules that imported it
    demo_working_dir = os.path.join(demo_dir, "demo_execution")
    os.makedirs(demo_working_dir, exist_ok=True)

    config.WORKING_DIR = demo_working_dir
    utils.WORKING_DIR = demo_working_dir
    features.WORKING_DIR = demo_working_dir
    tp.WORKING_DIR = demo_working_dir
    ip.WORKING_DIR = demo_working_dir

    # 2. Patch Data Paths in DataLoader
    dl.TRAIN_METADATA_PATH = paths["train_meta"]
    dl.VAL_METADATA_PATH = paths["val_meta"]
    dl.TEST_METADATA_PATH = paths["test_meta"]
    dl.TRAIN_TRACKING_PATH = paths["train_track"]
    dl.TEST_TRACKING_PATH = paths["test_track"]

    # 3. Patch Model Hyperparameters in ModelFactory for Speed
    fast_params = {
        "n_estimators": 2,
        "num_leaves": 4,
        "max_depth": 2,
        "n_jobs": 1,
        "random_state": 42,
        "verbose": -1,
    }

    # Update LGBM
    current_lgbm = mf.LGBM_PARAMS.copy()
    current_lgbm.update(fast_params)
    mf.LGBM_PARAMS = current_lgbm

    # Update XGB
    current_xgb = mf.XGB_PARAMS.copy()
    current_xgb.update(fast_params)
    # XGB specific adjustments
    if "num_leaves" in current_xgb:
        del current_xgb["num_leaves"]
    current_xgb["verbosity"] = 0
    mf.XGB_PARAMS = current_xgb

    # 4. Patch Submission Path
    demo_submission_path = os.path.join(demo_working_dir, "demo_submission.csv")
    config.SUBMISSION_PATH = demo_submission_path
    ip.SUBMISSION_PATH = demo_submission_path


if __name__ == "__main__":
    # Setup Demo Directory
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    try:
        # 1. Create Data
        paths = create_mini_datasets(DEMO_DIR)

        # 2. Patch Libraries
        patch_modules(paths, DEMO_DIR)

        # 3. Run Training Pipeline
        print("\n" + "=" * 30)
        print(" STARTING TRAINING PIPELINE ")
        print("=" * 30)
        # Force load_cached_data=False to ensure we process the mini datasets
        tp.TrainingPipeline.run_pipeline(load_cached_data=False)

        # 4. Run Inference Pipeline
        print("\n" + "=" * 30)
        print(" STARTING INFERENCE PIPELINE ")
        print("=" * 30)
        ip.InferencePipeline.run_inference(load_cached_data=False)

        # 5. Verification
        print("\n" + "=" * 30)
        print(" VERIFICATION ")
        print("=" * 30)

        submission_path = ip.SUBMISSION_PATH
        if not os.path.exists(submission_path):
            raise FileNotFoundError(f"Submission file not found at {submission_path}")

        df_sub = pd.read_csv(submission_path)
        print(f"Submission generated with {len(df_sub)} rows.")

        # Verify columns
        expected_cols = ["contact_id", "contact"]
        if not all(col in df_sub.columns for col in expected_cols):
            raise ValueError(
                f"Submission missing required columns. Found: {df_sub.columns}"
            )

        # Verify content (should match mini test metadata length)
        df_test_meta = pd.read_csv(paths["test_meta"])
        if len(df_sub) != len(df_test_meta):
            raise AssertionError(
                f"Submission length ({len(df_sub)}) does not match Test Metadata length ({len(df_test_meta)})"
            )

        print("Verification Successful! Demo completed.")

    except Exception as e:
        print(f"\nERROR: {e}")
        raise e
