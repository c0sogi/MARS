import os
import torch
import numpy as np
import pandas as pd
import shutil
import time

# Import library components
from library.config import Config
from library.utils import rle_encoding, calculate_fbeta, DiceBCELoss
from library.data import get_dataloaders
from library.model import StratifiedSegFormer
from library.train import train_model, set_seed
from library.inference import generate_submission


def run_demo():
    print("=== Starting Vesuvius Ink Detection Demo ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Set a separate working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set submission path
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Reduce training parameters
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce workers

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.NUM_EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # ------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test RLE Encoding
    # Create a simple binary mask: 0 0 1 1 1 0 0 1 0
    # Flattened indices (1-based): 3,4,5 and 8
    # Expected RLE: "3 3 8 1"
    dummy_mask = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
    # Flattened: 0 0 1 1 1 0 0 1 0
    rle_out = rle_encoding(dummy_mask)
    expected_rle = "3 3 8 1"
    assert (
        rle_out == expected_rle
    ), f"RLE check failed. Expected '{expected_rle}', got '{rle_out}'"
    print("RLE Encoding: OK")

    # Test F-beta Score
    # Pred: 0.8 (TP), 0.2 (TN), 0.9 (FP), 0.1 (FN) -> relative to Target 1, 0, 0, 1
    pred_t = torch.tensor([0.8, 0.2, 0.9, 0.1])
    target_t = torch.tensor([1.0, 0.0, 0.0, 1.0])
    # Threshold 0.5 -> Pred Bin: 1, 0, 1, 0
    # TP=1 (idx 0), FP=1 (idx 2), FN=1 (idx 3)
    # Precision = 1 / (1+1) = 0.5
    # Recall = 1 / (1+1) = 0.5
    # F0.5 = (1.25 * 0.5 * 0.5) / (0.25 * 0.5 + 0.5) = 0.3125 / 0.625 = 0.5
    score = calculate_fbeta(pred_t, target_t, beta=0.5, threshold=0.5)
    assert abs(score - 0.5) < 1e-5, f"F-beta check failed. Expected 0.5, got {score}"
    print("F-beta Calculation: OK")

    # ------------------------------------------------------------------------
    # 3. Data Loading & Processing
    # ------------------------------------------------------------------------
    print("\n[3] Initializing Data Loaders (Debug Mode)...")

    # Use debug=True to load a small subset of the data
    # This will also trigger MIP generation for the fragments involved in the subset
    start_time = time.time()
    train_loader, val_loader = get_dataloaders(load_cached_data=True, debug=True)
    print(f"Data Loaders initialized in {time.time() - start_time:.2f}s")

    # Verify Batch Shape
    # Fetch one batch
    images, masks = next(iter(train_loader))

    # Expected Image: (B, C, H, W) -> (4, 3, 512, 512)
    # Expected Mask: (B, 1, H, W) -> (4, 1, 512, 512)
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Mask Shape: {masks.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        Config.IN_CHANNELS,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect image batch shape"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect mask batch shape"
    print("Data Pipeline: OK")

    # ------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # ------------------------------------------------------------------------
    print("\n[4] Initializing Model and Checking Forward Pass...")

    device = torch.device(Config.DEVICE)
    model = StratifiedSegFormer(
        pretrained=False
    )  # Use False to avoid download delays/errors in demo
    model.to(device)

    # Move batch to device
    images = images.to(device)

    # Forward pass
    outputs = model(images)
    print(f"Model Output Shape: {outputs.shape}")

    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect model output shape"

    # Check Loss Function
    criterion = DiceBCELoss()
    masks = masks.to(device)
    loss = criterion(outputs, masks)
    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    print("Model & Loss: OK")

    # ------------------------------------------------------------------------
    # 5. Training Loop
    # ------------------------------------------------------------------------
    print("\n[5] Running Training Loop (Debug Mode)...")

    # train_model handles the loop, validation, and saving best_model.pth
    # We use debug=True to keep it fast
    train_model(debug=True)

    expected_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(expected_model_path), "Training failed to save best_model.pth"
    print(f"Training finished. Model saved to {expected_model_path}")

    # ------------------------------------------------------------------------
    # 6. Inference & Submission
    # ------------------------------------------------------------------------
    print("\n[6] Running Inference and Generating Submission...")

    # generate_submission uses the model in Config.WORKING_DIR/best_model.pth
    # and processes fragments listed in metadata/test.csv
    generate_submission(load_cached_data=True)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Validate submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(df_sub.head())

    assert (
        "Id" in df_sub.columns and "Predicted" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission file is empty"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
