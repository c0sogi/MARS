import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings
import logging

# Import library modules
# We import them all upfront to patch their configuration variables
import library.config as config
import library.utils as utils
import library.data_processing as data_processing
import library.feature_engineering as feature_engineering
import library.model_definitions as model_definitions
import library.mining_strategy as mining_strategy
import library.training_loop as training_loop
import library.inference as inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def setup_demo_environment():
    """
    Sets up a demo environment in ./working/demo_execution.
    Creates mini datasets and patches library configurations to use them.
    """
    print("Setting up demo environment...")

    # Define paths
    base_dir = "./working/demo_execution"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    # Original Paths
    orig_train_meta = "./metadata/train_metadata.csv"
    orig_val_meta = "./metadata/val_metadata.csv"
    orig_test_meta = "./metadata/test_metadata.csv"
    orig_train_track = "./input/train_player_tracking.csv"
    orig_test_track = "./input/test_player_tracking.csv"

    # Demo Paths
    demo_train_meta = os.path.join(base_dir, "mini_train_metadata.csv")
    demo_val_meta = os.path.join(base_dir, "mini_val_metadata.csv")
    demo_test_meta = os.path.join(base_dir, "mini_test_metadata.csv")
    demo_train_track = os.path.join(base_dir, "mini_train_tracking.csv")
    demo_test_track = os.path.join(base_dir, "mini_test_tracking.csv")
    demo_submission = os.path.join(base_dir, "demo_submission.csv")

    # 1. Create Mini Metadata
    # We take a small sample to ensure speed
    print("Creating mini datasets...")

    # Train Metadata (Sample 200 rows)
    df_tr = pd.read_csv(
        orig_train_meta, nrows=2000
    )  # Read a bit more to get enough unique plays
    df_tr_sample = df_tr.sample(n=200, random_state=42).reset_index(drop=True)
    df_tr_sample.to_csv(demo_train_meta, index=False)

    # Val Metadata (Sample 50 rows)
    df_val = pd.read_csv(orig_val_meta, nrows=1000)
    df_val_sample = df_val.sample(n=50, random_state=42).reset_index(drop=True)
    df_val_sample.to_csv(demo_val_meta, index=False)

    # Test Metadata (Sample 50 rows)
    df_test = pd.read_csv(orig_test_meta, nrows=1000)
    df_test_sample = df_test.sample(n=50, random_state=42).reset_index(drop=True)
    df_test_sample.to_csv(demo_test_meta, index=False)

    # 2. Create Mini Tracking Data
    # We must ensure the tracking data covers the game_plays in our metadata samples
    train_plays = set(df_tr_sample["game_play"]).union(set(df_val_sample["game_play"]))
    test_plays = set(df_test_sample["game_play"])

    # Load full tracking (it's around 150MB, manageable) and filter
    print("Filtering tracking data...")
    df_track_train_full = pd.read_csv(orig_train_track)
    df_track_train_mini = df_track_train_full[
        df_track_train_full["game_play"].isin(train_plays)
    ].copy()
    df_track_train_mini.to_csv(demo_train_track, index=False)

    df_track_test_full = pd.read_csv(orig_test_track)
    df_track_test_mini = df_track_test_full[
        df_track_test_full["game_play"].isin(test_plays)
    ].copy()
    df_track_test_mini.to_csv(demo_test_track, index=False)

    print(f"Mini Train Metadata: {len(df_tr_sample)} rows")
    print(f"Mini Train Tracking: {len(df_track_train_mini)} rows")

    # 3. Patch Library Modules
    # We need to update the constants in all imported modules to point to our demo files
    # and reduce model complexity for speed.

    print("Patching library configuration...")

    # Define new params for speed
    FAST_LGBM = config.LGBM_PARAMS.copy()
    FAST_LGBM.update({"n_estimators": 10, "num_leaves": 16})

    FAST_XGB = config.XGB_PARAMS.copy()
    FAST_XGB.update({"n_estimators": 10, "max_depth": 4})

    # Helper to patch a module
    def patch_module(mod):
        mod.TRAIN_METADATA_PATH = demo_train_meta
        mod.VAL_METADATA_PATH = demo_val_meta
        mod.TEST_METADATA_PATH = demo_test_meta
        mod.TRAIN_TRACKING_PATH = demo_train_track
        mod.TEST_TRACKING_PATH = demo_test_track
        mod.CACHE_DIR = base_dir
        mod.SUBMISSION_FILE = demo_submission

        # Patch params if they exist in the module
        if hasattr(mod, "LGBM_PARAMS"):
            mod.LGBM_PARAMS = FAST_LGBM
        if hasattr(mod, "XGB_PARAMS"):
            mod.XGB_PARAMS = FAST_XGB

    # Apply patches
    modules_to_patch = [
        config,
        utils,
        data_processing,
        feature_engineering,
        model_definitions,
        mining_strategy,
        training_loop,
        inference,
    ]

    for mod in modules_to_patch:
        patch_module(mod)

    # Also patch CacheManager in utils to ensure it picks up the new CACHE_DIR
    # (The class uses the global constant at instantiation, but we patched the module attr)

    return base_dir, demo_submission


def run_pipeline_demo():
    # 1. Setup
    utils.seed_everything(42)
    base_dir, submission_path = setup_demo_environment()

    # Configure logger to stdout
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("Demo")

    # 2. Run Training Loop
    print("\n" + "=" * 30)
    print("STARTING TRAINING LOOP DEMO")
    print("=" * 30)

    trainer = training_loop.TrainingLoop(logger=logger)

    # Force reload to ensure we use our mini datasets and don't pick up existing cache
    best_thresh, best_mcc = trainer.run_training(
        load_cached_data=False, load_cached_models=False
    )

    print(f"\nTraining complete. Threshold: {best_thresh:.4f}, MCC: {best_mcc:.4f}")

    # Verify Training Artifacts
    models_dir = os.path.join(base_dir, "models")
    assert os.path.exists(
        os.path.join(models_dir, "expert_lgbm.joblib")
    ), "LGBM model not saved"
    assert os.path.exists(
        os.path.join(models_dir, "expert_xgb.joblib")
    ), "XGB model not saved"
    assert os.path.exists(
        os.path.join(models_dir, "best_threshold.npy")
    ), "Threshold not saved"

    # 3. Run Inference Pipeline
    print("\n" + "=" * 30)
    print("STARTING INFERENCE PIPELINE DEMO")
    print("=" * 30)

    inferencer = inference.InferencePipeline(logger=logger)

    # Run inference
    df_submission = inferencer.run_inference(load_cached_data=False)

    # 4. Verify Submission
    print("\n" + "=" * 30)
    print("VERIFICATION")
    print("=" * 30)

    # Check file existence
    assert os.path.exists(submission_path), "Submission file was not created"

    # Check shape (should match mini test metadata size, which was 50)
    # Note: InferencePipeline saves the file, we reload to check
    df_check = pd.read_csv(submission_path)
    print(f"Submission shape: {df_check.shape}")

    assert len(df_check) == 50, f"Expected 50 predictions, got {len(df_check)}"
    assert "contact_id" in df_check.columns, "Missing contact_id column"
    assert "contact" in df_check.columns, "Missing contact column"
    assert df_check["contact"].isin([0, 1]).all(), "Predictions must be binary"

    print("SUCCESS: Pipeline executed and verified.")


if __name__ == "__main__":
    run_pipeline_demo()
