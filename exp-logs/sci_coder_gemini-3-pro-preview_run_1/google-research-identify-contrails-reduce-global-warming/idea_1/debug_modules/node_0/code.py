import os
import sys
import torch
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.dataset import ContrailDataset
from library.model import SimpleUNet
from library.train import ContrailLoss, train_model
from library.utils import rle_encode, GlobalDiceMetric, dice_coefficient
from library.inference import generate_submission


def run_demonstration():
    print("==== Starting Demonstration Script ====")

    # 1. Setup and Configuration
    print("\n[1] Setting up configuration and seeds...")
    Config.set_seed(42)

    # Verify directories
    assert os.path.exists(Config.INPUT_DIR), "Input directory missing"
    assert os.path.exists(Config.METADATA_DIR), "Metadata directory missing"

    # 2. Dataset Demonstration
    print("\n[2] Verifying Dataset logic...")
    # Instantiate training dataset with a small sample limit for speed
    max_samples_demo = 10
    train_ds = ContrailDataset(split="train", max_samples=max_samples_demo)

    # Check length
    print(f"    Dataset length: {len(train_ds)}")
    assert (
        len(train_ds) == max_samples_demo
    ), f"Expected {max_samples_demo} samples, got {len(train_ds)}"

    # Check item retrieval
    img, mask = train_ds[0]
    print(f"    Image shape: {img.shape}, Mask shape: {mask.shape}")

    # Verify shapes: Image should be (3, 256, 256), Mask (1, 256, 256)
    assert img.shape == (3, 256, 256), f"Incorrect image shape: {img.shape}"
    assert mask.shape == (1, 256, 256), f"Incorrect mask shape: {mask.shape}"

    # Verify normalization (Ash composite should be roughly in [0, 1])
    # Note: It's possible for some pixels to be slightly out if clipped, but logic ensures clip to 0,1
    assert img.min() >= 0.0 and img.max() <= 1.0, "Image data not normalized to [0, 1]"

    # Check Test Dataset instantiation
    test_ds = ContrailDataset(split="test", max_samples=max_samples_demo)
    t_img, t_mask = test_ds[0]
    # Test masks should be zeroed out placeholders
    assert torch.sum(t_mask) == 0, "Test mask should be empty"

    # 3. Model Architecture Demonstration
    print("\n[3] Verifying Model architecture...")
    model = SimpleUNet(in_channels=3, out_channels=1)

    # Create a dummy batch: (Batch_Size=2, Channels=3, H=256, W=256)
    dummy_input = torch.randn(2, 3, 256, 256)

    # Forward pass
    output = model(dummy_input)
    print(f"    Input shape: {dummy_input.shape} -> Output shape: {output.shape}")

    # Verify output shape (Batch, Out_Channels, H, W)
    assert output.shape == (2, 1, 256, 256), "Model output shape mismatch"

    # Verify output range (Sigmoid activation -> [0, 1])
    assert output.min() >= 0.0 and output.max() <= 1.0, "Model output not in [0, 1]"

    # 4. Loss and Metric Demonstration
    print("\n[4] Verifying Loss and Metrics...")
    criterion = ContrailLoss(bce_weight=0.5, dice_weight=0.5)

    # Create dummy predictions and targets
    # Preds: 0.8 where target is 1, 0.2 where target is 0
    dummy_target = torch.zeros(2, 1, 256, 256)
    dummy_target[:, :, 50:100, 50:100] = 1.0

    dummy_pred = torch.zeros(2, 1, 256, 256) + 0.2
    dummy_pred[:, :, 50:100, 50:100] = 0.8

    loss = criterion(dummy_pred, dummy_target)
    print(f"    Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # Global Dice Metric
    metric = GlobalDiceMetric()
    metric.update(dummy_pred, dummy_target, threshold=0.5)
    score = metric.compute()
    print(f"    Global Dice Score: {score:.4f}")
    # Since preds match targets perfectly after thresholding at 0.5
    # (0.8 > 0.5 -> 1, 0.2 < 0.5 -> 0), Dice should be 1.0
    assert abs(score - 1.0) < 1e-5, f"Expected Dice 1.0, got {score}"

    # 5. RLE Encoding Demonstration
    print("\n[5] Verifying RLE Encoding...")
    # Create a simple mask: 0 0 1 1 1 0 (flattened logic)
    # In Fortran order (column-major), this maps to pixels.
    # Let's make a 2x2 mask:
    # [[0, 1],
    #  [0, 1]]
    # Flattened F-order: (0,0), (1,0), (0,1), (1,1) -> 0, 0, 1, 1
    # Indices (1-based): 1  2  3  4
    # Values:            0  0  1  1
    # Run starts at 3, length 2.

    simple_mask = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    rle_str = rle_encode(simple_mask)
    print(f"    Mask:\n{simple_mask}")
    print(f"    RLE String: '{rle_str}'")

    # Expected: "3 2"
    assert rle_str == "3 2", f"RLE encoding failed. Expected '3 2', got '{rle_str}'"

    # Test empty mask
    empty_rle = rle_encode(np.zeros((10, 10)))
    assert empty_rle == "-", "Empty mask RLE should be '-'"

    # 6. Training Loop Demonstration
    print("\n[6] Running Training Loop (Truncated)...")
    # We use a very small subset and 1 epoch to verify the pipeline runs without error.
    # train_model saves the best model to Config.CHECKPOINT_DIR

    best_dice = train_model(
        epochs=1,
        batch_size=2,  # Small batch for demo
        debug=True,
        max_samples=20,  # Enough for a few batches
        patience=1,
    )

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."
    print(f"    Training finished. Best Dice: {best_dice:.4f}")

    # 7. Inference Demonstration
    print("\n[7] Running Inference Generation...")
    # Generate submission for a few test samples
    generate_submission(checkpoint_path=checkpoint_path, batch_size=2, max_samples=10)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    # Validate submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission shape: {df_sub.shape}")
    print(f"    Columns: {list(df_sub.columns)}")

    assert "record_id" in df_sub.columns, "Missing record_id column"
    assert "encoded_pixels" in df_sub.columns, "Missing encoded_pixels column"
    assert len(df_sub) == 10, f"Expected 10 predictions, got {len(df_sub)}"

    print("\n==== Demonstration Complete: All checks passed. ====")


if __name__ == "__main__":
    run_demonstration()
