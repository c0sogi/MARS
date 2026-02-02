import os
import shutil
import numpy as np
import pandas as pd
import torch
import glob

# Import provided library components
from library.config import Config
from library.data_loader import DataLoader
from library.features import FeatureEngineer
from library.trainer import Trainer
from library.inference import InferenceManager


def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_header(msg):
    """Helper to print formatted headers."""
    print(f"\n{'='*60}\n{msg}\n{'='*60}")


def run_demo():
    print_header("STARTING DEMO: NFL Player Contact Detection")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Isolation
    # -------------------------------------------------------------------------
    print("Configuring environment for demo run...")

    # Redirect cache and submission to a demo folder in working directory
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config attributes
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.MODEL_CHECKPOINT_PATH = os.path.join(DEMO_DIR, "best_model.pth")

    # Update cache paths based on new CACHE_DIR
    Config.CACHE_TRAIN_FEATURES = os.path.join(
        Config.CACHE_DIR, "train_features.parquet"
    )
    Config.CACHE_VAL_FEATURES = os.path.join(Config.CACHE_DIR, "val_features.parquet")
    Config.CACHE_TEST_FEATURES = os.path.join(Config.CACHE_DIR, "test_features.parquet")

    # Reduce training intensity for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 128  # Smaller batch size for small debug samples

    # Create necessary directories
    Config.setup_directories()
    set_seeds(Config.SEED)

    print(f"Cache Directory: {Config.CACHE_DIR}")
    print(f"Model Checkpoint: {Config.MODEL_CHECKPOINT_PATH}")
    print(f"Epochs: {Config.EPOCHS}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print_header("DEMO: Data Loading")

    # Load Metadata
    print("Loading Train Metadata...")
    df_meta_train = DataLoader.load_metadata(split="train")
    print(f"Train Metadata Shape: {df_meta_train.shape}")

    # Validate Metadata
    assert "game_play" in df_meta_train.columns
    assert "contact" in df_meta_train.columns
    assert len(df_meta_train) > 0

    # Filter for specific game_plays to demonstrate efficient loading
    sample_game_plays = df_meta_train["game_play"].unique()[:2].tolist()
    print(f"Loading Tracking Data for game_plays: {sample_game_plays}")

    df_tracking = DataLoader.load_tracking_data(
        split="train", game_plays=sample_game_plays, load_cached_data=False
    )
    print(f"Tracking Data Shape: {df_tracking.shape}")

    # Validate Tracking
    assert "x_position" in df_tracking.columns
    assert "speed" in df_tracking.columns
    assert df_tracking["game_play"].isin(sample_game_plays).all()

    print("Loading Helmets Data...")
    df_helmets = DataLoader.load_helmets_data(
        split="train", game_plays=sample_game_plays, load_cached_data=False
    )
    print(f"Helmets Data Shape: {df_helmets.shape}")

    # Validate Helmets
    assert "left" in df_helmets.columns
    assert "width" in df_helmets.columns

    # -------------------------------------------------------------------------
    # 3. Feature Engineering Demonstration
    # -------------------------------------------------------------------------
    print_header("DEMO: Feature Engineering")

    fe = FeatureEngineer()

    # Generate features for a small debug sample (e.g., 2000 rows)
    DEBUG_SAMPLE_SIZE = 2000
    print(f"Generating features for top {DEBUG_SAMPLE_SIZE} rows of training data...")

    df_features = fe.generate_features(
        split="train", load_cached_data=False, debug_sample=DEBUG_SAMPLE_SIZE
    )
    print(f"Generated Features Shape: {df_features.shape}")

    # Validate Feature Columns
    # 1. Check for Kinematic Lags (e.g., p1_x_position_t-5)
    expected_lag_col = f"p1_x_position_t-{Config.WINDOW_HALF}"
    assert (
        expected_lag_col in df_features.columns
    ), f"Missing lag column: {expected_lag_col}"

    # 2. Check for Visual Features
    assert "p1_vis_left" in df_features.columns, "Missing visual feature p1_vis_left"

    # 3. Check for Distance
    assert "dist" in df_features.columns, "Missing distance feature"

    # 4. Check Ground Imputation Logic (if 'G' is present in sample)
    ground_rows = df_features[df_features["nfl_player_id_2"] == "G"]
    if not ground_rows.empty:
        # For ground, p2 speed should be 0
        assert (
            ground_rows["p2_speed"] == 0
        ).all(), "Ground imputation for speed failed"
        print("Ground imputation verification passed.")

    # 5. Check Numerical Stability (Clamping)
    # Speed should be clamped between CLAMP_MIN and CLAMP_MAX
    # Note: speed is usually positive, so check max
    max_speed = df_features["p1_speed"].max()
    assert (
        max_speed <= Config.CLAMP_MAX
    ), f"Clamping failed. Max speed {max_speed} > {Config.CLAMP_MAX}"
    print("Feature generation and validation successful.")

    # -------------------------------------------------------------------------
    # 4. Training Demonstration
    # -------------------------------------------------------------------------
    print_header("DEMO: Model Training")

    trainer = Trainer()

    print("Starting training with debug_sample=2000 (1 Epoch)...")
    # Using a small sample to ensure it runs instantly
    trainer.train(debug_sample=2000)

    # Validate Outputs
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not created."
    assert trainer.best_threshold is not None, "Best threshold was not optimized."

    print(f"Training complete. Model saved to {Config.MODEL_CHECKPOINT_PATH}")
    print(f"Optimized Threshold: {trainer.best_threshold}")

    # -------------------------------------------------------------------------
    # 5. Inference Demonstration
    # -------------------------------------------------------------------------
    print_header("DEMO: Inference & Submission")

    inference_manager = InferenceManager()

    # 1. Initialize Resources (Refit scalers on training subset)
    print("Initializing inference resources (fitting scalers)...")
    inference_manager.initialize_resources(debug_sample=2000)

    # 2. Find Best Threshold (Optimize on validation subset)
    print("Optimizing threshold on validation subset...")
    inference_manager.find_best_threshold(debug_sample=1000)

    # 3. Generate Predictions (Full Test Set)
    # Note: We let this run on the full test set as required by the task structure.
    # The test set is ~460k rows, feature gen + inference should take < 2 mins.
    print("Generating predictions for test set...")
    inference_manager.generate_predictions()

    # Validate Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")
    print(f"Submission Columns: {df_sub.columns.tolist()}")

    assert "contact_id" in df_sub.columns
    assert "contact" in df_sub.columns
    assert df_sub["contact"].isin([0, 1]).all(), "Predictions must be binary (0/1)"

    # Check against sample submission length
    df_sample = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    assert len(df_sub) == len(
        df_sample
    ), f"Submission length mismatch. Expected {len(df_sample)}, got {len(df_sub)}"

    print("Submission validation passed.")
    print_header("DEMO COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_demo()
