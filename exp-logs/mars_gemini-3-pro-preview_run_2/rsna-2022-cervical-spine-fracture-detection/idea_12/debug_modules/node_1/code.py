import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_weighted_loss, get_logger
from library.data import get_dataloaders
from library.model import CalibratedSequenceModel
from library.engine import fit, inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    # --- 1. Setup & Configuration Override ---
    print("\n=== 1. Setup & Configuration ===")

    # Set seeds for reproducibility
    seed_everything(42)

    # Override Config for rapid demonstration (Speed Optimization)
    print("Overriding Config for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 12  # Small subset for quick iteration
    Config.EPOCHS = 1  # Only 1 epoch
    Config.SEQ_LEN = 8  # Reduced sequence length (from 96)
    Config.IMAGE_SIZE = (128, 128)  # Reduced resolution (from 384)
    Config.BATCH_SIZE = 2  # Small batch size
    Config.ACCUMULATION_STEPS = 1  # No accumulation needed for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Use a lighter backbone if possible for speed, otherwise keep default
    # We keep default B4 to ensure code compatibility, but image size reduction helps speed.

    # Ensure output directories exist
    Config.create_dirs()
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Image Size: {Config.IMAGE_SIZE}")
    print(f"Sequence Length: {Config.SEQ_LEN}")

    # --- 2. Data Loading Demonstration ---
    print("\n=== 2. Data Loading ===")

    # Initialize DataLoaders
    # This will use the metadata files in ./metadata and images in ./input
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Train Loader
    images, targets, uids = next(iter(train_loader))

    print(f"Batch Shapes:")
    print(f"  Images:  {images.shape} (Expected: [B, Seq, C, H, W])")
    print(f"  Targets: {targets.shape} (Expected: [B, 8])")

    # Assertions to verify logic
    assert images.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.IN_CHANNELS,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    ), f"Incorrect image batch shape: {images.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        8,
    ), f"Incorrect target batch shape: {targets.shape}"

    print("Data Loading verification successful.")

    # --- 3. Model Initialization & Forward Pass ---
    print("\n=== 3. Model Initialization ===")

    device = Config.DEVICE
    print(f"Device: {device}")

    model = CalibratedSequenceModel()
    model.to(device)

    # Dummy Forward Pass to check architecture compatibility
    print("Running dummy forward pass...")
    with torch.no_grad():
        dummy_input = images.to(device)
        output = model(dummy_input)

    print(f"Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (Config.BATCH_SIZE, 8), "Model output shape mismatch."
    assert torch.all(
        (output >= 0) & (output <= 1)
    ), "Model output probabilities out of range [0, 1]."

    print("Model initialization and forward pass successful.")

    # --- 4. Training Loop Demonstration ---
    print("\n=== 4. Training Loop (Fit) ===")

    # Run the fit function
    # This handles training, validation, and checkpointing
    trained_model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=Config.EPOCHS,
    )

    # Check if best model was saved
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Training complete. Checkpoint found at: {best_model_path}")
    else:
        # In a 1-epoch run with random init, validation might not improve over 'inf',
        # but the code logic should run without crashing.
        print(
            "Training complete (No checkpoint saved - metric might not have improved in 1 epoch)."
        )

    # --- 5. Inference Demonstration ---
    print("\n=== 5. Inference ===")

    # Run inference on the test set
    inference(trained_model, test_loader, device)

    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file was not created."

    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(sub_df)} rows.")
    print("First 5 rows:")
    print(sub_df.head())

    # Verify submission format
    assert (
        "row_id" in sub_df.columns and "fractured" in sub_df.columns
    ), "Submission columns missing."

    # --- 6. Metric Verification ---
    print("\n=== 6. Metric Verification ===")

    # Create dummy data to verify weighted loss logic
    # Case: Perfect prediction
    y_true = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]])
    y_pred_good = np.array(
        [
            [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            [0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        ]
    )

    # Case: Bad prediction
    y_pred_bad = np.array(
        [
            [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
            [0.99, 0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
        ]
    )

    loss_good = calculate_weighted_loss(y_true, y_pred_good)
    loss_bad = calculate_weighted_loss(y_true, y_pred_bad)

    print(f"Loss (Good Prediction): {loss_good:.4f}")
    print(f"Loss (Bad Prediction):  {loss_bad:.4f}")

    assert (
        loss_good < loss_bad
    ), "Loss function logic error: Good prediction should have lower loss."
    print("Metric verification successful.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
