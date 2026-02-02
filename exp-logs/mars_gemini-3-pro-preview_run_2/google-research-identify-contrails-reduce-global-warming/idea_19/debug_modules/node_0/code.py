import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, rle_encode, dice_coef_metric
from library.dataset import ContrailDataset, get_transforms
from library.model import AttentionGatedConvNeXtUNet
from library.loss import HybridBCEDiceLoss
from library.engine import train_model
from library.inference import run_inference


def main():
    print("============================================================")
    print("Running Contrail Identification Demo & Verification")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration Setup
    # ------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Set reproducible seed
    seed_everything(42)

    # Override Config for the demo to run quickly and safely
    # We use a separate working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.DEBUG = True  # Enable debug mode to sample dataset
    Config.DEBUG_SAMPLE_SIZE = 20  # Only use 20 samples for training/validation
    Config.NUM_WORKERS = 0  # Use main process to avoid overhead in demo

    # Create necessary directories
    Config.setup()
    Config.print_config()

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # ------------------------------------------------------------------
    # 2. Verify Utilities
    # ------------------------------------------------------------------
    print("\n[2] Verifying Utilities (RLE Encoding)...")

    # Test Case: 2x2 Identity Matrix
    # [[1, 0],
    #  [0, 1]]
    # Flattened (Fortran/Column-major): [1, 0, 0, 1]
    # Indices (1-based): 1, 4
    # Expected RLE: "1 1 4 1" (Start at 1 len 1, Start at 4 len 1)

    dummy_mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    encoded = rle_encode(dummy_mask)
    expected = "1 1 4 1"

    print(f"  Input Mask:\n{dummy_mask}")
    print(f"  Encoded: {encoded}")

    assert (
        encoded == expected
    ), f"RLE Encoding mismatch. Got {encoded}, expected {expected}"
    print("  > RLE Encoding verified successfully.")

    # ------------------------------------------------------------------
    # 3. Verify Dataset Pipeline
    # ------------------------------------------------------------------
    print("\n[3] Verifying Dataset Loading & Transforms...")

    # Initialize Train Dataset (Debug mode samples data)
    train_ds = ContrailDataset(
        split="train", transform=get_transforms("train"), debug=True
    )

    assert len(train_ds) > 0, "Train dataset is empty."
    print(f"  Train Dataset Size (Debug): {len(train_ds)}")

    # Fetch one sample
    sample = train_ds[0]
    img = sample["image"]
    mask = sample["mask"]
    rec_id = sample["record_id"]

    print(f"  Sample ID: {rec_id}")
    print(f"  Image Tensor Shape: {img.shape}")  # Should be (6, 256, 256)
    print(f"  Mask Tensor Shape: {mask.shape}")  # Should be (1, 256, 256)

    # Validations
    assert img.ndim == 3 and img.shape[0] == 6, f"Incorrect image shape: {img.shape}"
    assert mask.ndim == 3 and mask.shape[0] == 1, f"Incorrect mask shape: {mask.shape}"
    assert img.dtype == torch.float32, "Image tensor should be float32"
    assert mask.dtype == torch.float32, "Mask tensor should be float32"

    print("  > Dataset pipeline verified successfully.")

    # ------------------------------------------------------------------
    # 4. Verify Model Architecture
    # ------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = AttentionGatedConvNeXtUNet()
    model.to(device)

    # Create dummy input batch
    dummy_input = torch.randn(2, 6, 256, 256).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input Shape: {dummy_input.shape}")
    print(f"  Output Shape: {output.shape}")

    # Validations
    assert output.shape == (
        2,
        1,
        256,
        256,
    ), f"Model output shape mismatch: {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("  > Model architecture verified successfully.")

    # ------------------------------------------------------------------
    # 5. Verify Loss Function
    # ------------------------------------------------------------------
    print("\n[5] Verifying Loss Function...")

    criterion = HybridBCEDiceLoss()
    dummy_target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)

    loss = criterion(output, dummy_target)
    print(f"  Calculated Loss: {loss.item():.6f}")

    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("  > Loss function verified successfully.")

    # ------------------------------------------------------------------
    # 6. Verify Training Loop
    # ------------------------------------------------------------------
    print("\n[6] Verifying Training Loop (1 Epoch)...")

    # Prepare DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    val_ds = ContrailDataset(
        split="validation", transform=get_transforms("validation"), debug=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Execute Training
    # Note: train_model returns the model with best weights loaded
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        criterion=criterion,
        device=device,
        epochs=Config.EPOCHS,
    )

    # Verify Model Checkpoint
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print(f"  > Training loop completed. Model saved to {Config.BEST_MODEL_PATH}")

    # ------------------------------------------------------------------
    # 7. Verify Inference Pipeline
    # ------------------------------------------------------------------
    print("\n[7] Verifying Inference Pipeline...")

    # Run inference
    # Note: run_inference loads the model from Config.BEST_MODEL_PATH
    # It runs on the test set.

    try:
        run_inference(batch_size=Config.BATCH_SIZE, num_workers=0, device=device)
    except Exception as e:
        print(f"Inference failed with error: {e}")
        raise e

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission File Rows: {len(df_sub)}")
    print(f"  Columns: {list(df_sub.columns)}")

    assert len(df_sub) > 0, "Submission file is empty."
    assert "record_id" in df_sub.columns, "Missing 'record_id' column."
    assert "encoded_pixels" in df_sub.columns, "Missing 'encoded_pixels' column."

    # Check if there are any non-empty predictions
    non_empty = df_sub[df_sub["encoded_pixels"] != "-"].shape[0]
    print(f"  Non-empty predictions: {non_empty}")

    print("  > Inference pipeline verified successfully.")

    print("\n============================================================")
    print("All checks passed! The library is functional.")
    print("============================================================")


if __name__ == "__main__":
    main()
