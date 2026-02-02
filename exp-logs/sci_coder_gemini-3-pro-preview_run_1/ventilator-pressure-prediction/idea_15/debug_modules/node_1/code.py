import os
import shutil
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, ensure_dir
from library.features import FeatureEngineer
from library.dataset import get_data_loaders, VentilatorDataset
from library.model import VentilatorNet
from library.engine import Trainer


def main():
    print("=== Ventilator Pressure Prediction: Library Demo ===\n")

    # 1. Configuration Setup
    # We initialize Config with debug=True to use fewer epochs (2) and smaller batches.
    print("--- 1. Initializing Configuration ---")
    config = Config(debug=True)

    # Override working directory for this specific demo to avoid conflicts
    config.WORKING_DIR = "./working/demo_execution"
    ensure_dir(config.WORKING_DIR)

    # Update paths dependent on WORKING_DIR
    config.TRAIN_CACHE = os.path.join(config.WORKING_DIR, "train_engineered.parquet")
    config.VAL_CACHE = os.path.join(config.WORKING_DIR, "val_engineered.parquet")
    config.TEST_CACHE = os.path.join(config.WORKING_DIR, "test_engineered.parquet")
    config.SCALER_CENTER = os.path.join(config.WORKING_DIR, "scaler_center.npy")
    config.SCALER_SCALE = os.path.join(config.WORKING_DIR, "scaler_scale.npy")
    config.MODEL_PATH = os.path.join(config.WORKING_DIR, "model.pth")
    config.SUBMISSION_PATH = os.path.join(config.WORKING_DIR, "submission.csv")

    # Set random seeds for reproducibility
    seed_everything(config.SEED)
    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Device: {config.DEVICE}")

    # 2. Feature Engineering
    print("\n--- 2. Feature Engineering ---")
    fe = FeatureEngineer(config)

    # Run feature engineering in debug mode (samples 100 breaths per split)
    # We force load_cached_data=False to demonstrate the computation logic
    train_df, val_df, test_df = fe.run(load_cached_data=False, debug=True)

    # Verification: Check DataFrames
    print(f"Train DataFrame Shape: {train_df.shape}")
    print(f"Val DataFrame Shape: {val_df.shape}")

    # Verify engineered features exist
    expected_features = ["volume", "u_in_lag1", "u_in_diff1"]
    for feat in expected_features:
        assert feat in train_df.columns, f"Feature {feat} missing from Train DF"

    # Verify Scaler files were created
    assert os.path.exists(config.SCALER_CENTER), "Scaler center file not found"
    assert os.path.exists(config.SCALER_SCALE), "Scaler scale file not found"
    print("Feature Engineering verification passed.")

    # 3. Data Loading
    print("\n--- 3. Data Loading ---")
    # get_data_loaders handles tensor preparation and caching internally
    train_loader, val_loader, test_loader = get_data_loaders(config, debug=True)

    # Verification: Check Batch Structure
    batch = next(iter(train_loader))
    inputs = batch["input"]
    u_out = batch["u_out"]
    targets = batch["target"]

    print(f"Batch Input Shape: {inputs.shape}")  # Expected: (Batch, Seq, Features)
    print(f"Batch Target Shape: {targets.shape}")  # Expected: (Batch, Seq)

    assert (
        inputs.shape[1] == config.SEQ_LEN
    ), f"Sequence length mismatch. Got {inputs.shape[1]}, expected {config.SEQ_LEN}"
    assert (
        inputs.shape[2] == config.INPUT_DIM
    ), f"Feature dim mismatch. Got {inputs.shape[2]}, expected {config.INPUT_DIM}"
    assert u_out.shape == targets.shape, "u_out and target shapes must match"
    print("Data Loader verification passed.")

    # 4. Model Instantiation & Forward Pass
    print("\n--- 4. Model Architecture ---")
    model = VentilatorNet(config).to(config.DEVICE)

    # Verification: Forward pass with a dummy batch
    # Move batch to configured device
    inputs = inputs.to(config.DEVICE)
    u_out = u_out.to(config.DEVICE)
    targets = targets.to(config.DEVICE)

    output = model(inputs, u_out=u_out, target=targets)

    pred = output["prediction"]
    loss = output["loss"]

    print(f"Prediction Shape: {pred.shape}")
    print(f"Loss Value: {loss.item():.4f}")

    assert pred.shape == targets.shape, "Prediction shape mismatch"
    assert not torch.isnan(loss), "Loss is NaN"
    print("Model forward pass verification passed.")

    # 5. Training Loop (via Trainer)
    print("\n--- 5. Training Execution ---")
    # Re-initialize trainer to ensure clean state
    trainer = Trainer(config, debug=True)

    # Override the trainer's model with our verified model (optional, but good for consistency)
    trainer.model = model

    # Run training
    # In debug mode, this runs for config.EPOCHS (2) on the small subset
    trainer.fit()

    # Verification: Check if model checkpoint exists
    assert os.path.exists(
        config.MODEL_PATH
    ), "Model checkpoint not found after training"
    print("Training loop completed and model saved.")

    # 6. Inference & Submission
    print("\n--- 6. Inference Generation ---")
    trainer.generate_submission()

    # Verification: Check submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found"

    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission Shape: {sub_df.shape}")
    print(f"Submission Columns: {sub_df.columns.tolist()}")

    assert (
        "id" in sub_df.columns and "pressure" in sub_df.columns
    ), "Submission columns incorrect"
    assert len(sub_df) > 0, "Submission file is empty"

    # Check if predictions are not all zero (simple sanity check)
    assert (
        sub_df["pressure"].abs().sum() > 0
    ), "All predictions are zero, model might not have learned"

    print("Inference verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
