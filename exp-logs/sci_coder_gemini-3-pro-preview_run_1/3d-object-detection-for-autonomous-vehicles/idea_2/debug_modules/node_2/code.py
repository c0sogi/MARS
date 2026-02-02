import os
import torch
import numpy as np
import pandas as pd
import logging
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_logger
from library.dataset import BEVDataset
from library.model import DLASeg
from library.loss import CenterNetLoss
from library.train import Trainer
from library.inference import Predictor


def main():
    # 1. Setup and Configuration Overrides for Demo
    print(">>> Setting up configuration for fast demonstration...")

    # Set seed for reproducibility
    set_seed(42)

    # Configure logging to stdout
    logger = get_logger()

    # Override Config for speed and resource management
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2

    # Use a specific subdirectory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    Config.print_config()

    # 2. Dataset Verification
    print("\n>>> Verifying Dataset...")
    sample_size = 8
    dataset = BEVDataset(split="train", load_cached_data=True, sample_size=sample_size)

    print(f"Dataset loaded with {len(dataset)} samples.")
    assert (
        len(dataset) == sample_size
    ), f"Expected {sample_size} samples, got {len(dataset)}"

    # Fetch one sample
    sample = dataset[0]

    # Verify Input Shape: (Channels, Height, Width)
    # Config.INPUT_SIZE is (250, 250), Channels is 3
    expected_shape = (Config.IN_CHANNELS, Config.INPUT_SIZE[1], Config.INPUT_SIZE[0])
    input_tensor = sample["input"]
    print(f"Input tensor shape: {input_tensor.shape}")

    assert (
        input_tensor.shape == expected_shape
    ), f"Input shape mismatch. Expected {expected_shape}, got {input_tensor.shape}"

    # Verify Target Shapes
    # Heatmap: (NumClasses, H, W)
    hm = sample["hm"]
    expected_hm_shape = (Config.NUM_CLASSES, Config.INPUT_SIZE[1], Config.INPUT_SIZE[0])
    assert (
        hm.shape == expected_hm_shape
    ), f"Heatmap shape mismatch. Expected {expected_hm_shape}, got {hm.shape}"

    # Regression Mask: (MaxDetections,)
    mask = sample["mask"]
    assert mask.shape[0] == Config.MAX_DETECTIONS, "Mask shape mismatch"

    print("Dataset verification passed.")

    # 3. Model and Loss Verification
    print("\n>>> Verifying Model and Loss...")
    device = torch.device(Config.DEVICE)
    model = DLASeg().to(device)
    criterion = CenterNetLoss()

    # Create a dummy batch
    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0).to(device)
    targets_batch = {
        k: v.unsqueeze(0).to(device)
        for k, v in sample.items()
        if k != "input" and isinstance(v, torch.Tensor)
    }

    # Forward Pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_batch)

    # Verify Output Keys and Shapes
    required_heads = ["hm", "reg", "wh", "depth", "rot"]
    for head in required_heads:
        assert head in outputs, f"Model output missing head: {head}"
        # Output spatial dims should match input spatial dims (due to interpolation in model)
        assert outputs[head].shape[2:] == (
            Config.INPUT_SIZE[1],
            Config.INPUT_SIZE[0],
        ), f"Output spatial dim mismatch for head {head}"

    print("Model forward pass successful.")

    # Loss Calculation
    # We need gradients for loss verification usually, but here checking forward logic
    loss, loss_stats = criterion(outputs, targets_batch)
    print(f"Calculated Loss: {loss.item():.4f}")
    print(f"Loss Stats: {loss_stats}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss_stats["total_loss"] == loss.item(), "Total loss mismatch in stats"

    print("Loss verification passed.")

    # 4. Training Loop Demonstration
    print("\n>>> Running Training Loop (Demo)...")
    # Initialize Trainer with a small sample size
    trainer = Trainer(sample_size=16)

    # Run training (1 epoch as configured)
    trainer.train()

    # Verify model was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"

    print(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # 5. Inference Demonstration
    print("\n>>> Running Inference (Demo)...")
    # Initialize Predictor
    predictor = Predictor(checkpoint_path=Config.MODEL_SAVE_PATH)

    # Run inference on a small subset of test data
    predictor.run_inference(sample_size=10)

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file created with {len(sub_df)} rows.")

    assert (
        "Id" in sub_df.columns and "PredictionString" in sub_df.columns
    ), "Submission file missing required columns"

    # Check if we have any predictions (might be empty string if score threshold is high/model is untrained)
    # But the structure should be valid.
    # Since we trained for 1 epoch on 16 samples, the model is likely garbage,
    # but the pipeline should produce a valid CSV.

    print("Inference verification passed.")
    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
