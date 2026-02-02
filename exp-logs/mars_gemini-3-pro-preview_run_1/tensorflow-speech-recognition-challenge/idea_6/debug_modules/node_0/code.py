import os
import sys
import shutil
import random
import numpy as np
import pandas as pd
import torch
import warnings

# Import library components
from library.config import PathConfig, TrainConfig, ModelConfig, AudioConfig
from library.dataset import get_dataloaders
from library.model import FeaturePyramidEfficientNet
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_demo():
    print("Initializing Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # -------------------------------------------------------------------------
    # Enable Debug mode to use only 100 samples per dataset
    TrainConfig.DEBUG = True

    # Reduce training duration
    TrainConfig.EPOCHS = 2
    TrainConfig.EARLY_STOPPING_PATIENCE = 2

    # Use a specific working directory for this demo
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Monkey-patch PathConfig to point to the demo directory
    # Note: We must update derived paths manually as they were initialized at import
    PathConfig.WORKING_DIR = demo_dir
    PathConfig.MODEL_SAVE_PATH = os.path.join(demo_dir, "demo_best_model.pth")
    PathConfig.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")
    PathConfig.TRAIN_CACHE = os.path.join(demo_dir, "train_balanced.parquet")

    # Set seed for reproducibility
    set_seed(TrainConfig.SEED)

    print(f"Configuration updated. Output directory: {PathConfig.WORKING_DIR}")
    print(f"Debug Mode: {TrainConfig.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Data Loading and Verification
    # -------------------------------------------------------------------------
    print("\n[1/4] Verifying Data Loading...")

    # Generate dataloaders
    # load_cached_data=False ensures we regenerate the balanced parquet for this demo run
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch a single batch from training
    inputs, targets = next(iter(train_loader))

    # Verify Shapes
    # Expected Input: (Batch, Channels, Mels, Time)
    # Time dimension depends on AudioConfig.NUM_SAMPLES (16000) and HOP_LENGTH (160) -> approx 101 frames
    expected_time_dim = (AudioConfig.NUM_SAMPLES // AudioConfig.HOP_LENGTH) + 1

    print(f"Input batch shape: {inputs.shape}")
    print(f"Target batch shape: {targets.shape}")

    # Assert Input Dimensions
    assert inputs.dim() == 4, f"Expected 4D input tensor, got {inputs.dim()}"
    assert inputs.shape[1] == 1, f"Expected 1 channel (mono), got {inputs.shape[1]}"
    assert (
        inputs.shape[2] == AudioConfig.N_MELS
    ), f"Expected {AudioConfig.N_MELS} Mels, got {inputs.shape[2]}"
    # Allow small variance in time dimension due to padding/centering logic
    assert (
        abs(inputs.shape[3] - expected_time_dim) <= 2
    ), f"Unexpected time dimension: {inputs.shape[3]}"

    # Assert Target Dimensions and Values
    assert targets.dim() == 1, "Targets should be 1D"
    assert targets.max() < ModelConfig.NUM_CLASSES, "Target label index out of bounds"
    assert targets.min() >= 0, "Target label index negative"

    print("Data Loading Verification Passed.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[2/4] Verifying Model Architecture...")

    device = torch.device(TrainConfig.DEVICE)
    model = FeaturePyramidEfficientNet().to(device)

    # Move batch to device
    inputs = inputs.to(device)

    # Forward Pass
    model.eval()
    with torch.no_grad():
        logits = model(inputs)

    print(f"Logits shape: {logits.shape}")

    # Assert Output Dimensions
    assert logits.shape == (
        inputs.size(0),
        ModelConfig.NUM_CLASSES,
    ), f"Expected output shape {(inputs.size(0), ModelConfig.NUM_CLASSES)}, got {logits.shape}"

    print("Model Architecture Verification Passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[3/4] Executing Training Loop (2 Epochs)...")

    # Initialize Trainer
    trainer = Trainer()

    # Verify Trainer initialized with correct config overrides
    assert (
        trainer.model.training is False
    ), "Model should start in eval mode or default state"

    # Run Training
    # This will run for 2 epochs on the small debug dataset (100 samples)
    trainer.fit()

    # Verify Model Checkpoint Creation
    if not os.path.exists(PathConfig.MODEL_SAVE_PATH):
        # Note: If validation accuracy doesn't improve (unlikely with random init vs random chance),
        # the trainer might not save. However, the logic saves if val_acc > best_acc (init 0.0).
        # With 12 classes, random chance is ~8%.
        print(
            "Warning: No best model saved (Validation accuracy might not have exceeded 0.0)."
        )
        # Force save for submission testing
        torch.save(trainer.model.state_dict(), PathConfig.MODEL_SAVE_PATH)
    else:
        print(f"Checkpoint found at {PathConfig.MODEL_SAVE_PATH}")

    print("Training Loop Execution Passed.")

    # -------------------------------------------------------------------------
    # 5. Submission Generation Verification
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating Submission...")

    trainer.generate_submission()

    # Verify Submission File
    assert os.path.exists(PathConfig.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(PathConfig.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    # Assert Format
    assert list(df_sub.columns) == ["fname", "label"], "Submission columns mismatch"
    assert len(df_sub) > 0, "Submission file is empty"

    # In Debug mode, test set is also 100 samples
    if TrainConfig.DEBUG:
        assert len(df_sub) == 100, f"Expected 100 rows in debug mode, got {len(df_sub)}"

    print("Submission Generation Passed.")
    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
