import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings
import cv2

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, rle_encoding, fbeta_score, get_fragment_slab
from library.data import get_dataloaders, InkDataset
from library.model import SegFormerSpecialist
from library.train import train_specialist
from library.inference import predict_and_submit


def run_demo():
    # -------------------------------------------------------------------------
    # 0. Setup and Configuration Overrides
    # -------------------------------------------------------------------------
    print(">>> Step 0: Configuring environment for demo...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set a specific working directory for this demo to avoid conflicts
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override Config attributes for speed and demo purposes
    Config.WORKING_DIR = demo_working_dir
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.DEBUG = True  # Enable debug mode
    Config.MAX_TRAIN_SAMPLES = 4  # Limit to 4 samples to be very fast
    Config.NUM_WORKERS = 0  # Disable multiprocessing to avoid overhead in script
    Config.SUBMISSION_PATH = os.path.join(demo_working_dir, "submission.csv")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Batch Size: {Config.BATCH_SIZE}, Max Samples: {Config.MAX_TRAIN_SAMPLES}")

    # -------------------------------------------------------------------------
    # 1. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n>>> Step 1: Verifying Utility Functions...")

    # Test RLE Encoding
    # Create a simple mask: 0 1 1 0 1
    # Pixels (1-based): 2, 3 are 1s (start 2, len 2). 5 is 1 (start 5, len 1).
    dummy_mask = np.array([[0, 1, 1, 0, 1]], dtype=np.uint8)
    rle_str = rle_encoding(dummy_mask)
    expected_rle = "2 2 5 1"
    assert (
        rle_str == expected_rle
    ), f"RLE Encoding failed. Expected '{expected_rle}', got '{rle_str}'"
    print("RLE Encoding: OK")

    # Test F-Beta Score
    # Preds: 0.8 (Ink), 0.2 (No), 0.9 (Ink)
    # Target: 1 (Ink), 0 (No), 1 (Ink) -> Perfect match
    preds = torch.tensor([0.8, 0.2, 0.9])
    targets = torch.tensor([1.0, 0.0, 1.0])
    score = fbeta_score(preds, targets, beta=0.5, threshold=0.5)
    assert np.isclose(
        score, 1.0
    ), f"F-Beta Score logic failed. Expected 1.0, got {score}"
    print("F-Beta Score: OK")

    # -------------------------------------------------------------------------
    # 2. Demonstrate Data Loading & Slab Generation
    # -------------------------------------------------------------------------
    print("\n>>> Step 2: Demonstrating Data Loading...")

    # We will use the 'Mid' specialist configuration
    specialist_type = "Mid"

    # This function initializes the dataset and dataloaders.
    # It triggers `get_fragment_slab`, which computes MIPs from the TIF files.
    # Since we set MAX_TRAIN_SAMPLES=4, this should be quick after the slab is loaded.
    train_loader, val_loader = get_dataloaders(specialist_type)

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Verify Shapes
    # Image: (Batch, 3, 512, 512)
    # Label: (Batch, 1, 512, 512)
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        512,
        512,
    ), "Incorrect Image Tensor Shape"
    assert labels.shape == (
        Config.BATCH_SIZE,
        1,
        512,
        512,
    ), "Incorrect Label Tensor Shape"
    assert images.dtype == torch.float32, "Images should be float32"

    # Verify Cache Creation
    # The dataloader initialization should have created a cached .npy file in WORKING_DIR
    # We need to find the fragment ID used. The loader shuffles, so we check the directory.
    cached_files = [f for f in os.listdir(Config.WORKING_DIR) if f.endswith(".npy")]
    assert len(cached_files) > 0, "No cached slab files found in working directory."
    print(f"Cached files generated: {cached_files[:2]} ...")

    # -------------------------------------------------------------------------
    # 3. Demonstrate Model Architecture
    # -------------------------------------------------------------------------
    print("\n>>> Step 3: Demonstrating Model Architecture...")

    model = SegFormerSpecialist()
    model.to(Config.DEVICE)

    # Run a forward pass with the batch from Step 2
    images = images.to(Config.DEVICE)
    with torch.no_grad():
        logits = model(images)

    print(f"Output Logits Shape: {logits.shape}")

    # Verify Output Shape: (Batch, 1, 512, 512)
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
        512,
        512,
    ), "Model output shape mismatch"
    print("Forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Demonstrate Training Loop
    # -------------------------------------------------------------------------
    print("\n>>> Step 4: Demonstrating Training Loop (Specialist: Mid)...")

    # Train the 'Mid' specialist.
    # This will run for 1 epoch on 4 samples.
    # It should save 'model_Mid.pth' to Config.WORKING_DIR.
    best_score = train_specialist(
        specialist_type, epochs=Config.EPOCHS, gating_threshold=0.0
    )

    model_path = os.path.join(Config.WORKING_DIR, f"model_{specialist_type}.pth")
    assert os.path.exists(model_path), f"Model checkpoint not found at {model_path}"
    print(f"Training complete. Model saved to {model_path}")
    print(f"Best Val Score: {best_score}")

    # -------------------------------------------------------------------------
    # 5. Demonstrate Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n>>> Step 5: Demonstrating Inference Pipeline...")

    # predict_and_submit loads models for 'High', 'Mid', 'Low'.
    # We only trained 'Mid'. The inference script handles missing weights by
    # initializing random weights (with a warning). This is acceptable for the demo.

    predict_and_submit()

    # Verify Submission File
    submission_file = Config.SUBMISSION_PATH
    assert os.path.exists(submission_file), "Submission file was not generated."

    df_sub = pd.read_csv(submission_file)
    print("Submission File Head:")
    print(df_sub.head())

    assert (
        "Id" in df_sub.columns and "Predicted" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) > 0, "Submission file is empty."

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
