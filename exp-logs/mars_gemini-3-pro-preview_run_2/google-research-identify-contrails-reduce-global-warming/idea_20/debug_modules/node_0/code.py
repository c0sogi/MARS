import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library import config, utils, dataset, model as model_lib, loss, train, inference


def run_demo():
    print("===========================================================")
    print("       Contrail Identification Library Demo")
    print("===========================================================")

    # 1. Setup & Configuration Override
    # ----------------------------------------------------------------
    utils.seed_everything(42)

    # Define a specific output directory for this demo run
    demo_dir = os.path.join(config.WORKING_DIR, "demo_run")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override config paths to use the demo directory
    # Since modules import 'config', modifying attributes here affects them globally
    config.OUTPUT_DIR = demo_dir
    config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    print(f"Working Directory: {demo_dir}")

    # 2. Verify Utilities (RLE Encoding)
    # ----------------------------------------------------------------
    print("\n[1/6] Verifying RLE Encoding logic...")

    # Test Case 1: Single pixel at (0,0) in a 3x3 grid
    # Column-major indexing: (0,0) is index 1.
    mask_dot = np.zeros((3, 3), dtype=np.int8)
    mask_dot[0, 0] = 1
    rle_dot = utils.rle_encode(mask_dot)
    assert rle_dot == "1 1", f"RLE Error: Expected '1 1', got '{rle_dot}'"

    # Test Case 2: Vertical line at column 0, rows 0 and 1
    # Indices 1 and 2. Start at 1, length 2.
    mask_line = np.zeros((3, 3), dtype=np.int8)
    mask_line[0, 0] = 1
    mask_line[1, 0] = 1
    rle_line = utils.rle_encode(mask_line)
    assert rle_line == "1 2", f"RLE Error: Expected '1 2', got '{rle_line}'"

    # Test Case 3: Empty mask
    mask_empty = np.zeros((3, 3), dtype=np.int8)
    rle_empty = utils.rle_encode(mask_empty)
    assert rle_empty == "-", f"RLE Error: Expected '-', got '{rle_empty}'"

    print("  -> RLE Encoding verified successfully.")

    # 3. Verify Dataset and DataLoader
    # ----------------------------------------------------------------
    print("\n[2/6] Verifying Dataset and DataLoader...")

    # Instantiate DataLoader in debug mode (loads a small subset)
    # Using batch_size=4 for speed
    train_loader = dataset.get_dataloader(stage="train", batch_size=4, debug=True)

    # Fetch one batch
    try:
        images, masks = next(iter(train_loader))
        print(f"  -> Batch Shapes - Images: {images.shape}, Masks: {masks.shape}")

        # Assertions
        # Expected Image: (B, 6, 256, 256) -> 6 channels (3 Ash + 3 Diff)
        assert (
            images.dim() == 4 and images.shape[1] == 6
        ), f"Image shape mismatch. Expected (B, 6, H, W), got {images.shape}"
        assert (
            images.shape[2] == config.IMAGE_SIZE
            and images.shape[3] == config.IMAGE_SIZE
        ), "Image resolution mismatch."

        # Expected Mask: (B, 1, 256, 256)
        assert (
            masks.dim() == 4 and masks.shape[1] == 1
        ), f"Mask shape mismatch. Expected (B, 1, H, W), got {masks.shape}"

        print("  -> Dataset loaded and shapes verified.")

    except StopIteration:
        print("  -> Warning: Dataset is empty. Skipping shape verification.")

    # 4. Verify Model Architecture
    # ----------------------------------------------------------------
    print("\n[3/6] Verifying Model Architecture...")

    device = config.DEVICE
    # Initialize model (using non-pretrained for speed in demo, though config says True)
    model = model_lib.ConvNeXtUNet(
        backbone_name="convnext_tiny", pretrained=False, in_channels=6, num_classes=1
    ).to(device)

    # Create dummy input tensor (B=2, C=6, H=256, W=256)
    dummy_input = torch.randn(2, 6, 256, 256).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  -> Model Output Shape: {output.shape}")

    # Assert output shape matches (B, 1, H, W)
    assert output.shape == (
        2,
        1,
        256,
        256,
    ), f"Model output shape mismatch. Expected (2, 1, 256, 256), got {output.shape}"

    print("  -> Model forward pass verified.")

    # 5. Verify Loss Function
    # ----------------------------------------------------------------
    print("\n[4/6] Verifying Loss Function...")

    criterion = loss.HybridLoss()

    # Create dummy targets (binary 0 or 1)
    dummy_targets = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)

    # Calculate loss
    loss_val = criterion(output, dummy_targets)

    print(f"  -> Calculated Loss: {loss_val.item():.6f}")

    assert not torch.isnan(loss_val), "Loss returned NaN."
    assert loss_val >= 0, "Loss should be non-negative."

    print("  -> Loss function verified.")

    # 6. Run Training Loop (Integration Test)
    # ----------------------------------------------------------------
    print("\n[5/6] Running Training Loop (1 Epoch, Debug Mode)...")

    # Run training using the provided library function
    # debug=True ensures we use a small subset of data
    train.run_training(debug=True, epochs=1)

    # Check if model was saved
    model_path = os.path.join(config.OUTPUT_DIR, "best_model.pth")
    if os.path.exists(model_path):
        print(f"  -> Training complete. Best model saved at: {model_path}")
    else:
        print(
            "  -> Training complete. No model saved (Dice score likely 0.0 in first epoch)."
        )
        # Save the current model manually to allow inference step to proceed
        print("  -> Saving current model state for inference demonstration.")
        torch.save(model.state_dict(), model_path)

    # 7. Run Inference & Submission
    # ----------------------------------------------------------------
    print("\n[6/6] Running Inference and Generating Submission...")

    # Run inference using the trained (or dummy) model
    inference.make_submission(model_path=model_path, debug=True)

    submission_file = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Verify submission file exists and has correct format
    if os.path.exists(submission_file):
        df = pd.read_csv(submission_file)
        print(f"  -> Submission file generated with {len(df)} rows.")
        print(f"  -> Columns: {list(df.columns)}")

        assert "record_id" in df.columns, "Missing 'record_id' column."
        assert "encoded_pixels" in df.columns, "Missing 'encoded_pixels' column."

        # Check sample content
        print("  -> Sample rows:")
        print(df.head(3))
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n===========================================================")
    print("       Demo Completed Successfully")
    print("===========================================================")


if __name__ == "__main__":
    run_demo()
