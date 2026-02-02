import os
import pandas as pd
import numpy as np
import torch
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.train_eval import train_model, generate_submission
from library.models import RVCNet


def create_mini_dataset(demo_input_dir):
    """
    Creates a small subset of the data for demonstration purposes to ensure
    the script runs quickly.
    """
    print(f"Creating mini-dataset in {demo_input_dir}...")
    os.makedirs(demo_input_dir, exist_ok=True)

    # --- 1. Process Training Data ---
    # Load original metadata
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)

    # Select a small number of game_plays (e.g., 2)
    sample_game_plays = train_meta["game_play"].unique()[:2]
    mini_train_meta = train_meta[train_meta["game_play"].isin(sample_game_plays)].copy()

    # Save mini metadata
    mini_train_meta_path = os.path.join(demo_input_dir, "train.csv")
    mini_train_meta.to_csv(mini_train_meta_path, index=False)

    # Filter and save Tracking data
    # We read the full CSV but only keep rows for our sample plays
    # Note: Using iterator or reading full depending on memory.
    # Given the environment has 220GB RAM, reading full is fine.
    train_tracking = pd.read_csv(Config.TRAIN_TRACKING_PATH)
    mini_train_tracking = train_tracking[
        train_tracking["game_play"].isin(sample_game_plays)
    ].copy()
    mini_train_tracking_path = os.path.join(demo_input_dir, "train_player_tracking.csv")
    mini_train_tracking.to_csv(mini_train_tracking_path, index=False)

    # Filter and save Helmet data
    train_helmets = pd.read_csv(Config.TRAIN_HELMETS_PATH)
    mini_train_helmets = train_helmets[
        train_helmets["game_play"].isin(sample_game_plays)
    ].copy()
    mini_train_helmets_path = os.path.join(demo_input_dir, "train_baseline_helmets.csv")
    mini_train_helmets.to_csv(mini_train_helmets_path, index=False)

    # --- 2. Process Validation Data ---
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    sample_val_plays = val_meta["game_play"].unique()[:1]
    mini_val_meta = val_meta[val_meta["game_play"].isin(sample_val_plays)].copy()

    mini_val_meta_path = os.path.join(demo_input_dir, "validation.csv")
    mini_val_meta.to_csv(mini_val_meta_path, index=False)

    # Validation uses the same tracking/helmet files as train in the Config logic usually,
    # but strictly speaking, we need to ensure the tracking/helmet files we created above
    # ALSO contain the validation plays if they are from the same source file.
    # In the provided dataset, train_tracking covers all train/val plays.
    # So we append validation data to the mini tracking/helmet files.

    val_tracking = train_tracking[
        train_tracking["game_play"].isin(sample_val_plays)
    ].copy()
    val_helmets = train_helmets[
        train_helmets["game_play"].isin(sample_val_plays)
    ].copy()

    # Append to the files we just created
    val_tracking.to_csv(mini_train_tracking_path, mode="a", header=False, index=False)
    val_helmets.to_csv(mini_train_helmets_path, mode="a", header=False, index=False)

    # --- 3. Process Test Data ---
    test_meta = pd.read_csv(Config.TEST_META_PATH)
    # Extract game_play from contact_id if column doesn't exist (it should based on metadata script)
    sample_test_plays = test_meta["game_play"].unique()[:1]
    mini_test_meta = test_meta[test_meta["game_play"].isin(sample_test_plays)].copy()

    mini_test_meta_path = os.path.join(demo_input_dir, "test.csv")
    mini_test_meta.to_csv(mini_test_meta_path, index=False)

    test_tracking = pd.read_csv(Config.TEST_TRACKING_PATH)
    mini_test_tracking = test_tracking[
        test_tracking["game_play"].isin(sample_test_plays)
    ].copy()
    mini_test_tracking_path = os.path.join(demo_input_dir, "test_player_tracking.csv")
    mini_test_tracking.to_csv(mini_test_tracking_path, index=False)

    test_helmets = pd.read_csv(Config.TEST_HELMETS_PATH)
    mini_test_helmets = test_helmets[
        test_helmets["game_play"].isin(sample_test_plays)
    ].copy()
    mini_test_helmets_path = os.path.join(demo_input_dir, "test_baseline_helmets.csv")
    mini_test_helmets.to_csv(mini_test_helmets_path, index=False)

    print("Mini-dataset created successfully.")
    return {
        "train_meta": mini_train_meta_path,
        "val_meta": mini_val_meta_path,
        "test_meta": mini_test_meta_path,
        "train_tracking": mini_train_tracking_path,
        "train_helmets": mini_train_helmets_path,
        "test_tracking": mini_test_tracking_path,
        "test_helmets": mini_test_helmets_path,
    }


def run_demo():
    # 1. Setup
    seed_everything(42)

    # Define directories
    working_dir = "./working/demo_run"
    demo_data_dir = os.path.join(working_dir, "data")
    submission_dir = os.path.join(working_dir, "submission")

    # Clean up previous run if exists
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir)
    os.makedirs(submission_dir)

    # 2. Create Mini Dataset (Optimization for Speed)
    paths = create_mini_dataset(demo_data_dir)

    # 3. Patch Config to use Mini Dataset and Demo Settings
    print("Patching Config for demo run...")
    Config.WORKING_DIR = os.path.join(working_dir, "working")
    Config.SUBMISSION_DIR = submission_dir
    Config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Update Data Paths
    Config.TRAIN_META_PATH = paths["train_meta"]
    Config.VAL_META_PATH = paths["val_meta"]
    Config.TEST_META_PATH = paths["test_meta"]
    Config.TRAIN_TRACKING_PATH = paths["train_tracking"]
    Config.TRAIN_HELMETS_PATH = paths["train_helmets"]
    Config.TEST_TRACKING_PATH = paths["test_tracking"]
    Config.TEST_HELMETS_PATH = paths["test_helmets"]

    # Update Model Artifact Paths (since they depend on WORKING_DIR in class definition,
    # but we changed WORKING_DIR after class load, we must update these explicitly)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.SCALER_PATH = os.path.join(Config.WORKING_DIR, "scaler.joblib")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "rvc_net_model.pth")

    # Update Hyperparameters for Speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 64
    Config.NUM_WORKERS = 2
    Config.DEBUG = True

    # 4. Verify Model Architecture (Sanity Check)
    print("Verifying model architecture...")
    model = RVCNet()
    # Create dummy input based on config dimensions
    k_dim = Config.get_kinematic_input_dim()
    v_dim = Config.get_visual_input_dim()
    dummy_k = torch.randn(2, k_dim)
    dummy_v = torch.randn(2, v_dim)

    with torch.no_grad():
        out = model(dummy_k, dummy_v)

    assert out.shape == (2, 1), f"Expected output shape (2, 1), got {out.shape}"
    print("Model architecture verified.")

    # 5. Train Model
    # load_cached_data=False forces the pipeline to process our new mini-dataset
    print("Starting training pipeline...")
    best_threshold = train_model(load_cached_data=False)

    print(f"Training finished. Best threshold: {best_threshold}")

    # Verify artifacts exist
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved."
    assert os.path.exists(Config.SCALER_PATH), "Scaler file was not saved."
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "best_threshold.npy")
    ), "Threshold file not saved."

    # 6. Generate Submission
    print("Generating submission...")
    generate_submission(threshold=best_threshold)

    # 7. Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Check against test metadata length
    df_test_meta = pd.read_csv(paths["test_meta"])
    assert len(df_sub) == len(
        df_test_meta
    ), f"Submission row count mismatch. Expected {len(df_test_meta)}, got {len(df_sub)}"

    # Check columns
    assert (
        "contact_id" in df_sub.columns and "contact" in df_sub.columns
    ), "Submission missing required columns."

    # Check values
    assert (
        df_sub["contact"].isin([0, 1]).all()
    ), "Submission contains non-binary values."

    print("\n=== Demo Completed Successfully ===")
    print(f"Output available at: {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    run_demo()
