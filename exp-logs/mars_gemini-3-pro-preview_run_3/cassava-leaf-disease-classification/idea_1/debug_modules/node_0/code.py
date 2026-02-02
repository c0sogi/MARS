import os
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_dataset
from library.model import CassavaResNet
from library.trainer import run_training
from library.inference import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Cassava Leaf Disease Classification Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Overrides for Speed and Demonstration
    # ---------------------------------------------------------
    print("[1] Configuring environment for fast execution...")

    # Modify Config class attributes directly to control the behavior of library modules
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples for train/val/test
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.PRETRAINED = (
        False  # Disable downloading weights for speed/offline capability
    )

    # Set a custom working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "resnet18_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Create working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(42)
    print("    Configuration complete.")

    # ---------------------------------------------------------
    # 2. Dataset Component Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Dataset loading...")

    # Load training dataset in debug mode
    train_ds = get_dataset("train", debug=True)

    # Assertions
    assert (
        len(train_ds) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected dataset size {Config.DEBUG_SUBSET_SIZE}, got {len(train_ds)}"

    # Check item retrieval
    img, label = train_ds[0]

    # Verify Image Tensor: (Channels, Height, Width)
    assert img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Expected image shape (3, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE}), got {img.shape}"

    # Verify Label Tensor
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"

    print(f"    Dataset loaded successfully. Size: {len(train_ds)}")
    print(f"    Sample image shape: {img.shape}")

    # ---------------------------------------------------------
    # 3. Model Component Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model architecture...")

    device = get_device()
    model = CassavaResNet(pretrained=False, num_classes=Config.NUM_CLASSES)
    model.to(device)
    model.eval()

    # Create dummy input batch: (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Verify Output Shape: (Batch_Size, Num_Classes)
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected output shape (2, {Config.NUM_CLASSES}), got {output.shape}"

    print(f"    Model forward pass successful. Output shape: {output.shape}")

    # ---------------------------------------------------------
    # 4. Training Pipeline Execution
    # ---------------------------------------------------------
    print("\n[4] Executing Training Pipeline (1 Epoch)...")

    # Run the training loop provided in library/trainer.py
    run_training(
        debug=True,
        epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=1e-3,  # Slightly higher LR to try and move weights
    )

    # NOTE: The trainer only saves the model if validation accuracy improves.
    # In a 1-epoch debug run with random weights, this is not guaranteed.
    # We ensure the checkpoint exists for the inference step.
    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        print(
            "    Checkpoint not saved by trainer (low accuracy). Saving manual checkpoint for inference demo."
        )
        torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
    else:
        print("    Checkpoint saved by trainer.")

    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint file is missing!"

    # ---------------------------------------------------------
    # 5. Inference Pipeline Execution
    # ---------------------------------------------------------
    print("\n[5] Executing Inference Pipeline...")

    # Run the inference loop provided in library/inference.py
    run_inference(
        checkpoint_path=Config.MODEL_CHECKPOINT_PATH,
        output_path=Config.SUBMISSION_PATH,
        debug=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated!"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Verify Columns
    expected_cols = ["image_id", "label"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(df_sub.columns)}"

    # Verify Length (Should match DEBUG_SUBSET_SIZE)
    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} predictions, got {len(df_sub)}"

    print(f"    Submission generated successfully at {Config.SUBMISSION_PATH}")
    print(f"    First 3 rows:\n{df_sub.head(3)}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
