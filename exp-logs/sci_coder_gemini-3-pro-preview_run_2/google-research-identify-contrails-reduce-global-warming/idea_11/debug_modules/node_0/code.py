import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, rle_encode
from library.dataset import ContrailDataset
from library.model import ConvNeXtUNet
from library.loss import HybridLoss
from library.engine import train_model, predict_and_submit


def run_demo():
    print("=== Starting Contrail Identification Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config parameters for a fast run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 12  # Small subset for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Redirect outputs to a specific demo directory in working
    Config.IDEA_NAME = "demo_run"
    Config.WORKING_DIR = os.path.join("./working", Config.IDEA_NAME)
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Re-initialize directories
    Config.setup_directories()

    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Dataset & DataLoader Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Dataset and DataLoader...")

    # Initialize Train Dataset
    train_ds = ContrailDataset(split="train", debug=True)
    print(f"    Train Dataset Size: {len(train_ds)}")

    # Validate Dataset Length
    if len(train_ds) != Config.DEBUG_SAMPLE_SIZE:
        raise AssertionError(
            f"Expected {Config.DEBUG_SAMPLE_SIZE} samples, got {len(train_ds)}"
        )

    # Fetch a single sample to verify shapes and normalization
    img, mask = train_ds[0]
    print(f"    Sample Image Shape: {img.shape}")
    print(f"    Sample Mask Shape: {mask.shape}")

    # Assertions for shapes (C, H, W)
    if img.shape != (Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE):
        raise AssertionError(f"Incorrect image shape: {img.shape}")
    if mask.shape != (1, Config.IMG_SIZE, Config.IMG_SIZE):
        raise AssertionError(f"Incorrect mask shape: {mask.shape}")

    # Assertions for Data Range (Normalization to [0, 1])
    if img.min() < 0.0 or img.max() > 1.0:
        raise AssertionError(
            f"Image data out of range [0, 1]: min={img.min()}, max={img.max()}"
        )

    # Initialize DataLoaders
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_ds = ContrailDataset(split="validation", debug=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    print("    DataLoaders initialized successfully.")

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = ConvNeXtUNet()
    model.to(device)

    # Create a dummy batch matching the config
    dummy_input = torch.randn(
        Config.BATCH_SIZE, Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE
    ).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")

    # Assert Output Shape (B, 1, H, W)
    expected_shape = (Config.BATCH_SIZE, 1, Config.IMG_SIZE, Config.IMG_SIZE)
    if output.shape != expected_shape:
        raise AssertionError(
            f"Expected output shape {expected_shape}, got {output.shape}"
        )

    # Check for NaNs
    if torch.isnan(output).any():
        raise AssertionError("Model output contains NaNs.")
    print("    Forward pass successful.")

    # ---------------------------------------------------------
    # 4. Loss Function Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Loss Function...")

    loss_fn = HybridLoss()
    dummy_target = (
        torch.randint(0, 2, (Config.BATCH_SIZE, 1, Config.IMG_SIZE, Config.IMG_SIZE))
        .float()
        .to(device)
    )

    loss = loss_fn(output, dummy_target)
    print(f"    Calculated Loss: {loss.item():.4f}")

    if torch.isnan(loss) or loss.item() < 0:
        raise AssertionError("Loss is NaN or negative.")
    print("    Loss calculation successful.")

    # ---------------------------------------------------------
    # 5. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop Demo (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Train for one epoch
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        loss_fn,
        num_epochs=Config.EPOCHS,
    )

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise AssertionError(f"Checkpoint not found at {best_model_path}")
    print("    Training loop completed and checkpoint saved.")

    # ---------------------------------------------------------
    # 6. RLE Encoding Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying RLE Encoding Logic...")

    # Create a synthetic 2x2 mask
    # [[0, 1],
    #  [0, 1]]
    # Flattened (Fortran/Column-major): 0, 0, 1, 1
    # Indices (1-based): 1, 2, 3, 4
    # Run: Start at 3, Length 2 -> "3 2"
    test_mask = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    rle_result = rle_encode(test_mask)
    print(f"    RLE Result: '{rle_result}'")

    if rle_result != "3 2":
        raise AssertionError(f"RLE Encoding failed. Expected '3 2', got '{rle_result}'")
    print("    RLE encoding verified.")

    # ---------------------------------------------------------
    # 7. Inference & Submission
    # ---------------------------------------------------------
    print("\n[7] Generating Submission...")

    test_ds = ContrailDataset(split="test", debug=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    predict_and_submit(
        trained_model,
        test_loader,
        device,
        threshold=0.5,
        use_tta=False,
        output_path=Config.SUBMISSION_PATH,
    )

    # Verify CSV
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not created.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission Rows: {len(df_sub)}")

    if len(df_sub) != len(test_ds):
        raise AssertionError(
            f"Submission row count mismatch. Expected {len(test_ds)}, got {len(df_sub)}"
        )

    required_cols = ["record_id", "encoded_pixels"]
    if not all(col in df_sub.columns for col in required_cols):
        raise AssertionError(f"Submission missing required columns: {required_cols}")

    print("    Submission generated and verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
