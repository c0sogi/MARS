import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders
from library.model import SiameseMultiScaleDiffNet
from library.engine import fit
from library.inference import generate_submission


def main():
    print("=== SETI Signal Detection Pipeline Demo ===")

    # 1. Setup & Configuration Override
    # We override Config parameters to make this demo run fast
    seed_everything(42)
    device = Config.DEVICE
    print(f"Device: {device}")

    # Modify Config for speed/demo purposes
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.DEBUG_SAMPLE_SIZE = 32  # Use only 32 samples for training/val
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # 2. Data Loading Demonstration
    print("\n[1/4] Testing Data Loading...")
    # get_loaders(debug=True) samples a small subset based on Config.DEBUG_SAMPLE_SIZE
    train_loader, val_loader = get_loaders(debug=True)

    # Fetch one batch to verify shapes
    batch_data, batch_target = next(iter(train_loader))
    stream_a = batch_data["stream_a"]
    stream_b = batch_data["stream_b"]

    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Stream A Shape: {stream_a.shape}")  # Expected: (B, 3, 288, 256)
    print(f"  Stream B Shape: {stream_b.shape}")  # Expected: (B, 3, 288, 256)
    print(f"  Target Shape:   {batch_target.shape}")

    # Verify Dimensions (Padding logic check)
    # Original height is 273, padded to 288 (divisible by 32)
    assert stream_a.shape == (Config.BATCH_SIZE, 3, 288, 256), "Stream A shape mismatch"
    assert stream_b.shape == (Config.BATCH_SIZE, 3, 288, 256), "Stream B shape mismatch"
    assert batch_target.shape == (Config.BATCH_SIZE,), "Target shape mismatch"
    print("  Data Loading verification passed.")

    # 3. Model Instantiation & Forward Pass
    print("\n[2/4] Testing Model Architecture...")
    model = SiameseMultiScaleDiffNet().to(device)

    # Perform a dummy forward pass
    with torch.no_grad():
        # Move inputs to device
        input_a = stream_a.to(device)
        input_b = stream_b.to(device)
        output = model(input_a, input_b)

    print(f"  Model Output Shape: {output.shape}")

    # Verify output shape (Batch_Size, 1)
    assert output.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    print("  Model architecture verification passed.")

    # 4. Training Loop Demonstration
    print("\n[3/4] Testing Training Loop (1 Epoch)...")

    # Setup Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX
    )

    # Run training
    # This uses library.engine.fit
    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=1,
    )

    # Verify model checkpoint creation
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    assert os.path.exists(
        best_model_path
    ), f"Model checkpoint not found at {best_model_path}"
    print("  Training loop verification passed. Best model saved.")

    # 5. Inference Demonstration
    print("\n[4/4] Testing Inference Pipeline (TTA)...")

    # Create a dummy test CSV with a few samples to avoid running inference on the full 6000 set
    original_test_csv = Config.TEST_CSV
    dummy_test_path = os.path.join(Config.OUTPUT_DIR, "dummy_test.csv")

    # Read first 10 rows of actual test metadata to create a valid dummy file
    df_test_full = pd.read_csv(original_test_csv)
    df_test_dummy = df_test_full.head(10).copy()
    df_test_dummy.to_csv(dummy_test_path, index=False)

    # Temporarily point Config to the dummy test file
    Config.TEST_CSV = dummy_test_path

    submission_path = os.path.join(Config.OUTPUT_DIR, "submission_demo.csv")

    # Run Inference
    # This uses library.inference.generate_submission which handles TTA
    generate_submission(model, device, output_path=submission_path)

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file not generated"
    df_sub = pd.read_csv(submission_path)
    print(f"  Submission Head:\n{df_sub.head(3)}")

    assert len(df_sub) == 10, f"Expected 10 predictions, got {len(df_sub)}"
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission columns missing"
    print("  Inference verification passed.")

    # Cleanup
    Config.TEST_CSV = original_test_csv  # Restore config
    if os.path.exists(dummy_test_path):
        os.remove(dummy_test_path)

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
