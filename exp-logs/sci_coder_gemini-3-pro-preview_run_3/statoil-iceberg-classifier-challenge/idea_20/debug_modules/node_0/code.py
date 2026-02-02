import os
import sys
import shutil
import numpy as np
import torch

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_loaders
from library.model import SelectiveSECNN
from library.trainer import train_fold


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup
    # --------------------------------------------------------------------------
    print("Configuring environment for fast execution...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SIZE = 20  # Use only 20 samples for verification
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size

    # Set a specific working directory for this demo to avoid cache conflicts
    Config.WORKING_DIR = "./working/demo_usage"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")

    # Ensure directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Set random seed
    seed_everything(42)

    # --------------------------------------------------------------------------
    # 2. Data Loader Verification
    # --------------------------------------------------------------------------
    print("\n[Step 1] Verifying Data Loaders...")

    # Force reload to ensure we use the DEBUG_SIZE and new working dir
    train_loader, val_loader, test_loader, ids_test = get_loaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=False
    )

    print(f"  Train Loader Batches: {len(train_loader)}")
    print(f"  Val Loader Batches:   {len(val_loader)}")
    print(f"  Test Loader Batches:  {len(test_loader)}")

    # Fetch one batch to verify shapes
    images, angles, labels = next(iter(train_loader))

    print(
        f"  Batch Shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions
    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75), "Image batch shape mismatch!"
    assert angles.shape == (Config.BATCH_SIZE,), "Angle batch shape mismatch!"
    assert labels.shape == (Config.BATCH_SIZE,), "Label batch shape mismatch!"
    assert images.dtype == torch.float32, "Images must be FloatTensor"
    assert angles.dtype == torch.float32, "Angles must be FloatTensor"

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[Step 2] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    print(f"  Using device: {device}")

    model = SelectiveSECNN().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(images, angles)

    print(f"  Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (Config.BATCH_SIZE,), "Model output shape mismatch!"
    assert not torch.isnan(logits).any(), "Model produced NaN outputs!"

    # --------------------------------------------------------------------------
    # 4. Training Loop Verification
    # --------------------------------------------------------------------------
    print("\n[Step 3] Verifying Training Loop (Fold 0)...")

    # Run training for one fold (2 epochs as configured)
    trained_model, best_score = train_fold(0, train_loader, val_loader)

    print(f"  Training finished. Best Validation Score (LogLoss): {best_score:.4f}")

    # Assertions
    assert isinstance(
        trained_model, SelectiveSECNN
    ), "train_fold did not return a model instance"
    assert isinstance(best_score, float), "best_score is not a float"
    assert best_score > 0, "LogLoss must be positive"

    # Verify checkpoint creation
    expected_ckpt = os.path.join(Config.CHECKPOINT_DIR, "model_best_fold_0.pth")
    assert os.path.exists(
        expected_ckpt
    ), f"Checkpoint file not found at {expected_ckpt}"

    # --------------------------------------------------------------------------
    # 5. Inference Verification
    # --------------------------------------------------------------------------
    print("\n[Step 4] Verifying Inference...")

    trained_model.eval()
    all_probs = []

    with torch.no_grad():
        for i, (test_images, test_angles) in enumerate(test_loader):
            test_images = test_images.to(device)
            test_angles = test_angles.to(device)

            out = trained_model(test_images, test_angles)
            probs = torch.sigmoid(out)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs)

    print(f"  Total Predictions: {len(all_probs)}")
    print(f"  First 5 Probabilities: {all_probs[:5]}")

    # Assertions
    assert len(all_probs) == len(
        ids_test
    ), "Number of predictions does not match test set size"
    assert np.all(
        (all_probs >= 0) & (all_probs <= 1)
    ), "Probabilities must be in [0, 1]"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
