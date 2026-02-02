import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, dice_score, rle_encode
from library.dataset import ContrailDataset
from library.model import ResNet34UNet
from library.loss import CombinedLoss
from library.train import run_training
from library.predict import generate_submission

if __name__ == "__main__":
    print("==== Starting Contrail Identification Pipeline Demonstration ====")

    # --------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # --------------------------------------------------------------------------
    # Set seeds for reproducibility
    set_seed(42)

    # Override Config parameters for a fast demonstration
    print("Configuring parameters for rapid execution...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Use 0 workers to avoid multiprocessing overhead in demo
    Config.DEBUG_SUBSET_SIZE = 12  # Use a tiny subset of data

    # Ensure working directory is clean for this run (optional, but good for demo)
    # We rely on Config.WORKING_DIR creation in config.py, which is already imported.

    # --------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # --------------------------------------------------------------------------
    print("\n[1/5] Verifying Utility Functions...")

    # Test RLE Encoding
    # Create a 10x10 mask with a vertical line of 3 pixels starting at (0,0)
    # In column-major flattening (Fortran), indices are 1, 2, 3.
    # Expected RLE: "1 3" (start at 1, length 3)
    dummy_mask = np.zeros((10, 10), dtype=np.uint8)
    dummy_mask[0:3, 0] = 1
    rle_result = rle_encode(dummy_mask)
    print(f"RLE Test Result: '{rle_result}'")
    assert (
        rle_result == "1 3"
    ), f"RLE Encoding failed. Expected '1 3', got '{rle_result}'"

    # Test Dice Score
    # Perfect match should be 1.0
    t1 = torch.ones((1, 1, 10, 10))
    t2 = torch.ones((1, 1, 10, 10))
    score = dice_score(t1, t2)
    assert np.isclose(score, 1.0), f"Dice Score failed. Expected 1.0, got {score}"
    print("Utility functions verified.")

    # --------------------------------------------------------------------------
    # 3. Verify Dataset Loading
    # --------------------------------------------------------------------------
    print("\n[2/5] Verifying Dataset Pipeline...")

    # Initialize dataset with debug subset
    train_ds = ContrailDataset(
        split="train", debug_subset_size=Config.DEBUG_SUBSET_SIZE
    )
    print(f"Dataset initialized with {len(train_ds)} samples.")

    # Fetch one sample
    image, mask = train_ds[0]

    # Check shapes
    # Image: (3, 256, 256) - 3 channels (Ash composite)
    # Mask: (1, 256, 256)
    print(f"Sample Image Shape: {image.shape}")
    print(f"Sample Mask Shape: {mask.shape}")

    assert image.shape == (3, 256, 256), f"Incorrect image shape: {image.shape}"
    assert mask.shape == (1, 256, 256), f"Incorrect mask shape: {mask.shape}"

    # Check value ranges
    assert image.min() >= 0.0 and image.max() <= 1.0, "Image values out of range [0, 1]"
    unique_mask_vals = torch.unique(mask)
    assert all(val in [0.0, 1.0] for val in unique_mask_vals), "Mask is not binary"
    print("Dataset pipeline verified.")

    # --------------------------------------------------------------------------
    # 4. Verify Model and Loss
    # --------------------------------------------------------------------------
    print("\n[3/5] Verifying Model and Loss...")

    # Initialize model
    # Using pretrained=False for speed in this unit test check (train loop uses True)
    model = ResNet34UNet(in_channels=3, out_channels=1, pretrained=False)
    model.eval()

    # Forward pass with the sample image (add batch dimension)
    input_tensor = image.unsqueeze(0)  # (1, 3, 256, 256)
    with torch.no_grad():
        output = model(input_tensor)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (1, 1, 256, 256), f"Incorrect output shape: {output.shape}"

    # Verify Loss
    criterion = CombinedLoss()
    # Create a dummy target with batch dimension
    target_tensor = mask.unsqueeze(0)
    loss = criterion(output, target_tensor)

    print(f"Calculated Loss: {loss.item():.4f}")
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss is NaN"
    print("Model and Loss verified.")

    # --------------------------------------------------------------------------
    # 5. Execute Training Loop
    # --------------------------------------------------------------------------
    print("\n[4/5] Executing Training Loop (1 Epoch)...")

    # Run training
    # This will save 'best_model.pth' to Config.CHECKPOINT_DIR
    run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        debug_subset_size=Config.DEBUG_SUBSET_SIZE,
        early_stopping_patience=1,
    )

    expected_checkpoint = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(expected_checkpoint), "Checkpoint file was not created."
    print("Training loop completed successfully.")

    # --------------------------------------------------------------------------
    # 6. Execute Inference and Generate Submission
    # --------------------------------------------------------------------------
    print("\n[5/5] Executing Inference...")

    # Generate submission using the trained model
    submission_df = generate_submission(
        checkpoint_path=expected_checkpoint,
        batch_size=Config.BATCH_SIZE,
        device=Config.DEVICE,
    )

    # Verify submission content
    assert not submission_df.empty, "Submission DataFrame is empty"
    assert "record_id" in submission_df.columns, "Missing 'record_id' column"
    assert "encoded_pixels" in submission_df.columns, "Missing 'encoded_pixels' column"

    # Verify file existence
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission CSV file was not saved."

    print(
        f"Inference complete. Submission generated with {len(submission_df)} records."
    )
    print("\n==== Demonstration Complete: All Systems Go ====")
