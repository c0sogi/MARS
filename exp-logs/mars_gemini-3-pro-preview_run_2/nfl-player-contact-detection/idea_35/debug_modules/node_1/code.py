import os
import sys
import shutil
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.feature_engineering import impute_ground, prepare_features
from library.dataset import get_train_val_datasets, ContactDataset
from library.model import APIRVNet, InputClamping
from library.loss import BinaryFocalLoss
from library.train import train_model
from library.inference import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_config():
    """
    Overrides Config parameters for a fast, isolated demo run.
    """
    print(">>> Setting up demo configuration...")

    # Create a specific working directory for this execution
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Update Config paths to point to the demo directory
    Config.WORKING_DIR = demo_dir
    Config.CACHE_TRAIN_FEATURES = os.path.join(demo_dir, "train_features.parquet")
    Config.CACHE_VAL_FEATURES = os.path.join(demo_dir, "val_features.parquet")
    Config.CACHE_TEST_FEATURES = os.path.join(demo_dir, "test_features.parquet")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SCALER_SAVE_PATH = os.path.join(demo_dir, "scaler.joblib")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Optimize hyperparameters for speed
    Config.MAX_EPOCHS = 1
    Config.BATCH_SIZE = 64  # Small batch for debug
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    Config.EARLY_STOPPING_PATIENCE = 1

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Max Epochs: {Config.MAX_EPOCHS}")


def test_input_clamping():
    """
    Verifies the InputClamping layer logic.
    """
    print("\n>>> Testing InputClamping Layer...")
    min_val, max_val = -5.0, 5.0
    layer = InputClamping(min_val, max_val)

    # Create data with outliers
    data = torch.tensor([-10.0, 0.0, 10.0])
    output = layer(data)

    # Check bounds
    assert torch.all(output >= min_val), "InputClamping failed lower bound check"
    assert torch.all(output <= max_val), "InputClamping failed upper bound check"
    assert output[0] == min_val, "Lower outlier not clamped correctly"
    assert output[2] == max_val, "Upper outlier not clamped correctly"
    print("    InputClamping logic verified.")


def test_impute_ground_logic():
    """
    Verifies the ground imputation logic manually.
    """
    print("\n>>> Testing Ground Imputation Logic...")

    # Construct a DataFrame mimicking the feature structure
    # Cite debug_lesson_23: Dynamically Derive Expected Dimensions from Configuration
    data = {
        "nfl_player_id_2": ["G", "12345"],
        "left_2": [100.0, 100.0],  # Visual feature
    }

    # Generate all expected kinematic columns based on Config
    kin_feats = [
        "x_position",
        "y_position",
        "speed",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]
    start_lag = -Config.WINDOW_LAG
    end_lag = Config.WINDOW_FUTURE

    for shift_step in range(start_lag, end_lag + 1):
        suffix = f"_t{shift_step:+d}" if shift_step != 0 else ""
        for feat in kin_feats:
            data[f"{feat}{suffix}_1"] = [10.0, 20.0]
            data[f"{feat}{suffix}_2"] = [0.0, 5.0]
            if feat == "speed":
                data[f"{feat}{suffix}_2"] = [5.0, 5.0]

    df = pd.DataFrame(data)

    # Run imputation
    df_imputed = impute_ground(df)

    # Check Row 0 (Ground) at t0 (suffix "")
    # Position P2 should match P1
    assert (
        df_imputed.loc[0, "x_position_2"] == df_imputed.loc[0, "x_position_1"]
    ), "Ground imputation failed: P2 position did not match P1."
    # Speed P2 should be 0
    assert (
        df_imputed.loc[0, "speed_2"] == 0.0
    ), "Ground imputation failed: P2 speed not zeroed."
    # Visual P2 should be 0
    assert (
        df_imputed.loc[0, "left_2"] == 0.0
    ), "Ground imputation failed: P2 visual feature not zeroed."

    # Check Row 1 (Player) - Should be untouched
    assert (
        df_imputed.loc[1, "x_position_2"] == 5.0
    ), "Ground imputation incorrect: Non-ground player modified."

    print("    Ground imputation logic verified.")


def run_pipeline_demo():
    """
    Runs the full training and inference pipeline using debug mode.
    """
    print("\n>>> Starting Pipeline Execution (Debug Mode)...")

    # 1. Train Model
    # debug=True samples 5000 rows from metadata
    print("    Step 1: Training Model...")
    model, best_mcc = train_model(debug=True)

    # Validation
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    assert os.path.exists(Config.SCALER_SAVE_PATH), "Scaler file was not saved."
    assert isinstance(best_mcc, float), "Best MCC is not a float."
    print(f"    Training complete. Best MCC: {best_mcc:.4f}")

    # 2. Run Inference
    # debug=True samples data for inference as well
    print("\n    Step 2: Running Inference...")
    run_inference(debug=True)

    # Validation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert "contact_id" in df_sub.columns, "Submission missing contact_id column."
    assert "contact" in df_sub.columns, "Submission missing contact column."
    assert not df_sub.empty, "Submission file is empty."

    # Check binary predictions
    unique_vals = df_sub["contact"].unique()
    assert all(val in [0, 1] for val in unique_vals), "Predictions are not binary."

    print(f"    Inference complete. Submission generated with {len(df_sub)} rows.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()

    # 2. Unit Tests
    test_input_clamping()
    test_impute_ground_logic()

    # 3. Integration Test (Full Pipeline)
    run_pipeline_demo()

    print("\n>>> Demo Execution Successfully Completed.")
