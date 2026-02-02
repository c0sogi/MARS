import os
import json
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_fold_loaders, get_test_loader, load_raw_data
from library.model import PPCWBN
from library.train_eval import run_fold


def run_demo():
    print("Starting Demo Execution...")

    # ==========================================
    # 1. CONFIGURATION OVERRIDE
    # ==========================================
    # Modify Config to run a fast, lightweight demo
    print("\n[1] Configuring environment for demo...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Small subset for speed
    Config.NUM_EPOCHS = 2  # Only 2 epochs to prove the loop works
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Use a specific directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_PATH = os.path.join(
        Config.WORKING_DIR, "cache", "processed_data_debug.npz"
    )
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.CACHE_PATH), exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    print(f"  Working Directory: {Config.WORKING_DIR}")
    print(f"  Debug Mode: {Config.DEBUG}")

    # ==========================================
    # 2. DATA LOADING VERIFICATION
    # ==========================================
    print("\n[2] Verifying Data Loading...")

    # This will trigger data processing and caching if not present
    # Using fold 0 for demonstration
    train_loader, val_loader, scaling_stats, angle_mean = get_fold_loaders(
        fold=0, load_cached_data=False
    )

    print("  Loaders initialized.")

    # Fetch one batch to verify shapes
    images, angles, labels = next(iter(train_loader))

    print(f"  Image Batch Shape: {images.shape}")
    print(f"  Angle Batch Shape: {angles.shape}")
    print(f"  Label Batch Shape: {labels.shape}")

    # Assertions
    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75), "Incorrect image batch shape"
    assert angles.shape == (Config.BATCH_SIZE,), "Incorrect angle batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"
    assert not torch.isnan(
        angles
    ).any(), "NaNs found in incidence angles after imputation"

    print("  Data Loading verification passed.")

    # ==========================================
    # 3. MODEL ARCHITECTURE VERIFICATION
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = PPCWBN().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    logits = model(images, angles)

    print(f"  Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"

    print("  Model verification passed.")

    # ==========================================
    # 4. TRAINING PIPELINE EXECUTION
    # ==========================================
    print("\n[4] Executing Training Pipeline (Fold 0)...")

    # run_fold handles the training loop, validation, and saving
    result = run_fold(fold_idx=0)

    print("  Training finished.")
    print(f"  Result: {result}")

    # Verify artifacts exist
    assert os.path.exists(result["model_path"]), "Model checkpoint not found"
    assert os.path.exists(result["stats_path"]), "Scaling stats file not found"

    print("  Training pipeline verification passed.")

    # ==========================================
    # 5. INFERENCE & SUBMISSION
    # ==========================================
    print("\n[5] Running Inference and Generating Submission...")

    # Load scaling stats
    with open(result["stats_path"], "r") as f:
        stats_data = json.load(f)
        saved_stats = stats_data["scaling_stats"]
        saved_angle_mean = stats_data["angle_mean"]

    # Get Test Loader
    test_loader, test_ids = get_test_loader(
        saved_stats, saved_angle_mean, load_cached_data=True
    )

    # Load best model
    model = PPCWBN().to(device)
    model.load_state_dict(torch.load(result["model_path"], map_location=device))
    model.eval()

    predictions = []

    with torch.no_grad():
        for i, (images, angles) in enumerate(test_loader):
            images = images.to(device)
            angles = angles.to(device)

            logits = model(images, angles)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()

            predictions.extend(probs)

            # Limit inference for demo speed
            if i >= 2:
                break

    # Since we broke early, we slice ids to match predictions length
    # In a real run, we would iterate the full loader
    limit_len = len(predictions)
    demo_ids = test_ids[:limit_len]

    # Create DataFrame
    submission_df = pd.DataFrame({"id": demo_ids, "is_iceberg": predictions})

    print(f"  Generated {len(submission_df)} predictions.")
    print("  Sample predictions:")
    print(submission_df.head())

    # Save Submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Assertions
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    assert submission_df["is_iceberg"].min() >= 0.0, "Probabilities < 0 found"
    assert submission_df["is_iceberg"].max() <= 1.0, "Probabilities > 1 found"

    print("  Inference and Submission verification passed.")
    print("\nDemo Execution Completed Successfully.")


if __name__ == "__main__":
    run_demo()
