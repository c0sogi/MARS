import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import BBSLNet
from library.train import train_model, generate_submission, LaplaceLoss


def run_demo():
    print("=== Starting BBSL-Net Demo ===\n")

    # 1. Setup & Reproducibility
    seed_everything(42)

    # 2. Patch Configuration for Fast Demonstration
    # The library uses static class attributes for configuration.
    # We override them to ensure the code runs within the time/compute constraints of a demo.
    print("[Config] Patching configuration for fast execution...")
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple debugging
    Config.IDEA_ID = "demo_run"

    # Redirect outputs to a demo-specific working directory
    Config.WORKING_DIR = os.path.join("./working", Config.IDEA_ID)
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure these directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print(f"  > Epochs: {Config.EPOCHS}")
    print(f"  > Batch Size: {Config.BATCH_SIZE}")
    print(f"  > Working Dir: {Config.WORKING_DIR}")

    # 3. Data Loading Verification
    print("\n[Data] Verifying Data Loading...")
    # debug=True triggers the logic in get_dataloaders to take a small subset of data
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch a single batch to inspect structure
    batch = next(iter(train_loader))

    # Verify keys
    expected_keys = ["image_axial", "image_coronal", "tabular", "target", "meta"]
    for key in expected_keys:
        if key not in batch:
            raise AssertionError(f"Missing key '{key}' in data batch.")

    # Verify Tensor Shapes
    # Image: (Batch, 3, 224, 224)
    img_shape = batch["image_axial"].shape
    if img_shape != (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE):
        raise AssertionError(f"Incorrect image shape: {img_shape}")

    # Tabular: (Batch, 8) -> [Weeks, Age, Sex, Smoke0, Smoke1, Smoke2, BaseFVC, BasePct]
    tab_shape = batch["tabular"].shape
    if tab_shape != (Config.BATCH_SIZE, 8):
        raise AssertionError(f"Incorrect tabular shape: {tab_shape}")

    print("  > Data batch structure and shapes verified.")

    # 4. Model Architecture Verification
    print("\n[Model] Verifying Architecture and Forward Pass...")
    device = Config.DEVICE
    model = BBSLNet().to(device)

    # Move inputs to device
    img_ax = batch["image_axial"].to(device)
    img_cor = batch["image_coronal"].to(device)
    tabular = batch["tabular"].to(device)

    # Perform Forward Pass
    # Returns: alpha (slope), sigma_base (intercept uncertainty), sigma_growth (slope uncertainty)
    alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

    # Verify Output Shapes: (Batch,)
    if alpha.shape != (Config.BATCH_SIZE,):
        raise AssertionError(f"Output 'alpha' shape mismatch: {alpha.shape}")

    # Verify Constraints
    # Sigma values come from Softplus, must be positive
    if torch.any(sigma_base <= 0) or torch.any(sigma_growth <= 0):
        raise AssertionError("Model produced non-positive uncertainty values.")

    print("  > Forward pass successful. Output shapes and constraints verified.")

    # 5. Loss Function Verification
    print("\n[Loss] Verifying Laplace Loss Calculation...")
    criterion = LaplaceLoss()
    target = batch["target"].to(device)
    base_fvc = batch["meta"]["Base_FVC"].to(device).float()
    weeks = batch["meta"]["Weeks"].to(device).float()

    # Reconstruct FVC and Confidence from parameters
    fvc_pred = base_fvc + alpha * weeks
    sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

    # Calculate Loss
    loss = criterion(target, fvc_pred, sigma_pred)

    # Check validity
    if torch.isnan(loss) or torch.isinf(loss):
        raise AssertionError("Loss calculation resulted in NaN or Inf.")

    print(f"  > Loss calculated successfully: {loss.item():.4f}")

    # 6. Training Loop Execution
    print("\n[Train] Executing Training Loop (Debug Mode)...")
    # This runs the full training routine but with reduced epochs/data due to our patches
    best_model_path = train_model(debug=True)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Expected model checkpoint at {best_model_path} not found."
        )

    print(f"  > Training complete. Best model saved at: {best_model_path}")

    # 7. Inference & Submission
    print("\n[Inference] Generating Submission...")
    generate_submission(best_model_path, debug=True)

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Validate Submission Content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    if sub_df.empty:
        raise AssertionError("Generated submission file is empty.")

    required_cols = ["Patient_Week", "FVC", "Confidence"]
    if not all(col in sub_df.columns for col in required_cols):
        raise AssertionError(
            f"Submission missing required columns. Found: {sub_df.columns}"
        )

    print("  > Submission file generated and validated.")
    print(sub_df.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
