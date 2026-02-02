import os
import shutil
import numpy as np
import torch

# Import functions and classes from the provided library
from library.utils import set_seed, get_device
from library.model import CAFPCNN
from library.data import get_loaders
from library.train import fit_fold


def main():
    print("=== Iceberg Classification Library Demo ===\n")

    # ---------------------------------------------------------
    # 1. Utilities and Setup
    # ---------------------------------------------------------
    print("[1/4] Testing Utilities...")

    # Set seed for reproducibility
    set_seed(42)

    # Check device availability
    device = get_device()
    print(f"   Device selected: {device}")

    # Verify seed works (simple numpy check)
    rand_check = np.random.rand()
    print(f"   Random check (seeded): {rand_check:.6f}")

    print("   Utilities verified.\n")

    # ---------------------------------------------------------
    # 2. Model Architecture Verification
    # ---------------------------------------------------------
    print("[2/4] Testing CAFPCNN Model Architecture...")

    # Instantiate model
    model = CAFPCNN().to(device)

    # Create dummy inputs
    # Batch size: 4, Channels: 3 (Band1, Band2, Avg), Height/Width: 75
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 75, 75).to(device)
    # Incidence angles
    dummy_angles = torch.tensor([35.0, 40.5, 30.2, 45.0]).to(device)

    # Perform forward pass
    model.eval()
    with torch.no_grad():
        logits = model(dummy_images, dummy_angles)

    # Check output dimensions
    print(f"   Input shape: {dummy_images.shape}")
    print(f"   Output shape: {logits.shape}")

    if logits.shape != (batch_size, 1):
        raise AssertionError(
            f"Expected output shape {(batch_size, 1)}, got {logits.shape}"
        )

    print("   Model forward pass successful.\n")

    # ---------------------------------------------------------
    # 3. Data Pipeline Verification
    # ---------------------------------------------------------
    print("[3/4] Testing Data Pipeline...")

    # Initialize loaders
    # We use fold 0 of 5. load_cached_data=True will generate cache if missing.
    train_loader, val_loader, test_loader = get_loaders(
        fold=0,
        n_folds=5,
        batch_size=8,
        num_workers=0,  # Use 0 workers for simple debugging/demo to avoid overhead
    )

    # Fetch one batch from training loader
    images, angles, labels = next(iter(train_loader))

    print(f"   Batch Images: {images.shape}")
    print(f"   Batch Angles: {angles.shape}")
    print(f"   Batch Labels: {labels.shape}")

    # Validate Data Shapes
    # Expected: (Batch, 3, 75, 75)
    if images.shape[1] != 3:
        raise AssertionError("Data loader did not produce 3-channel images.")
    if images.shape[2:] != (75, 75):
        raise AssertionError("Data loader produced incorrect image resolution.")

    # Validate Data Types
    if images.dtype != torch.float32:
        raise AssertionError("Images should be float32.")

    print("   Data pipeline verified.\n")

    # ---------------------------------------------------------
    # 4. Training Integration Test (Short Run)
    # ---------------------------------------------------------
    print("[4/4] Testing Training Loop Integration...")

    # Define a temporary directory for demo checkpoints
    demo_ckpt_dir = "./working/demo_usage/checkpoints"
    if os.path.exists(demo_ckpt_dir):
        shutil.rmtree(demo_ckpt_dir)
    os.makedirs(demo_ckpt_dir, exist_ok=True)

    # Run fit_fold with minimal parameters for speed
    # This verifies the connection between data, model, optimizer, and loss.
    best_loss = fit_fold(
        fold=0,
        n_folds=5,
        epochs=1,  # Run only 1 epoch for demonstration
        patience=1,
        batch_size=16,
        learning_rate=1e-3,
        save_dir=demo_ckpt_dir,
    )

    print(f"   Training run complete. Best Validation Loss: {best_loss:.4f}")

    # Verify that the model checkpoint was saved
    expected_ckpt_path = os.path.join(demo_ckpt_dir, "model_fold_0.pth")
    if not os.path.exists(expected_ckpt_path):
        raise AssertionError(f"Checkpoint file was not created at {expected_ckpt_path}")

    print(f"   Checkpoint verified at: {expected_ckpt_path}")
    print("   Training integration verified.\n")

    print("=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    main()
