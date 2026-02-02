import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd

# Import library modules
# Note: We import Config first to override settings before other modules use them
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_processing import DataProcessor, GestureDataset, get_dataloaders
from library.model import VIARN
from library.loss import CascadedLoss
from library.trainer import Trainer
from library.inference import run_inference, InferenceEngine


def main():
    print("=== Starting VI-ARN Library Demonstration ===")

    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print("\n[Step 1] Configuring Environment for Demo...")

    # Define a separate working directory for this demo to avoid conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config parameters for speed and isolation
    Config.WORKING_DIR = demo_dir
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "outputs", "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Cache paths
    os.makedirs(os.path.join(demo_dir, "cache"), exist_ok=True)
    Config.CACHE_TRAIN_PATH = os.path.join(demo_dir, "cache", "dataset_train.npz")
    Config.CACHE_VAL_PATH = os.path.join(demo_dir, "cache", "dataset_val.npz")
    Config.CACHE_TEST_PATH = os.path.join(demo_dir, "cache", "dataset_test.npz")

    # Runtime parameters
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SUBSET_SIZE = 10  # Only use 10 samples

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated. Running in:", demo_dir)

    # ==========================================
    # 2. Data Processing Demonstration
    # ==========================================
    print("\n[Step 2] Demonstrating Data Processing...")

    processor = DataProcessor()

    # Process training subset
    print("Processing training subset...")
    features, labels, boundaries = processor.process_dataset(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_PATH,
        is_train=True,
        debug_size=Config.DEBUG_SUBSET_SIZE,
        load_cached=False,
    )

    # Validation
    print(f"Features Shape: {features.shape}")
    print(f"Labels Shape: {labels.shape}")
    print(f"Boundaries: {boundaries}")

    # Assertions
    assert features.ndim == 2, "Features should be 2D (TotalFrames, FeatureDim)"
    assert labels.ndim == 1, "Labels should be 1D (TotalFrames,)"
    assert (
        features.shape[0] == labels.shape[0]
    ), "Feature and Label frame counts mismatch"
    # Feature dim = 180 (Kinematics) + 13 (Audio) = 193
    assert (
        features.shape[1] == 193
    ), f"Expected feature dim 193, got {features.shape[1]}"
    assert (
        len(boundaries) == Config.DEBUG_SUBSET_SIZE + 1
    ), "Incorrect number of sequence boundaries"

    print("Data Processing logic verified.")

    # ==========================================
    # 3. Model Architecture Demonstration
    # ==========================================
    print("\n[Step 3] Demonstrating Model Architecture...")

    model = VIARN()
    model.eval()

    # Create dummy input: (Batch, Time, InputDim)
    batch_size = 2
    time_steps = 64
    input_dim = 193
    dummy_input = torch.randn(batch_size, time_steps, input_dim)

    # Forward pass
    outputs = model(dummy_input)

    # Validation
    assert isinstance(outputs, dict), "Model output should be a dictionary"
    assert (
        "stage1" in outputs and "stage2" in outputs and "stage3" in outputs
    ), "Missing stages in output"

    stage3_out = outputs["stage3"]
    print(f"Stage 3 Output Shape: {stage3_out.shape}")

    # Output shape should be (Batch, Time, NumClasses)
    # NumClasses = 21 (20 gestures + background)
    expected_shape = (batch_size, time_steps, 21)
    assert (
        stage3_out.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {stage3_out.shape}"

    print("Model architecture verified.")

    # ==========================================
    # 4. Loss Function Demonstration
    # ==========================================
    print("\n[Step 4] Demonstrating Loss Function...")

    criterion = CascadedLoss()

    # Create dummy targets: (Batch, Time)
    dummy_targets = torch.randint(0, 21, (batch_size, time_steps), dtype=torch.long)

    # Compute loss
    loss, metrics = criterion(outputs, dummy_targets)

    print(f"Total Loss: {loss.item():.4f}")
    print("Metrics:", metrics.keys())

    # Validation
    assert loss.dim() == 0, "Loss should be a scalar"
    assert "total_loss" in metrics
    assert "stage3_loss" in metrics

    print("Loss function verified.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n[Step 5] Demonstrating Training Loop...")

    # Initialize Trainer
    trainer = Trainer()

    # Run fit (Training + Validation)
    # This uses the subset size defined in Config.DEBUG_SUBSET_SIZE
    trainer.fit(debug_subset=Config.DEBUG_SUBSET_SIZE)

    # Verify model checkpoint exists
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created!")

    # ==========================================
    # 6. Inference Demonstration
    # ==========================================
    print("\n[Step 6] Demonstrating Inference Pipeline...")

    # Initialize Inference Engine
    # It loads the model we just trained
    engine = InferenceEngine()

    # Generate submission for a subset of test data
    engine.generate_submission(debug_subset=Config.DEBUG_SUBSET_SIZE)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission generated at {Config.SUBMISSION_PATH}")

        # Check content
        with open(Config.SUBMISSION_PATH, "r") as f:
            lines = f.readlines()
            print(f"Submission has {len(lines)} lines.")
            if len(lines) > 0:
                print("First line example:", lines[0].strip())

            # We expect DEBUG_SUBSET_SIZE lines
            assert (
                len(lines) == Config.DEBUG_SUBSET_SIZE
            ), f"Expected {Config.DEBUG_SUBSET_SIZE} predictions, got {len(lines)}"
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
