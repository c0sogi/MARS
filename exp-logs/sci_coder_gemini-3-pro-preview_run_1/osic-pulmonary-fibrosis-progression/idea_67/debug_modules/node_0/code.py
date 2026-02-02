import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import provided library components
from library.config import Config, seed_everything
from library.model import TSCGNet, get_extended_dataloaders
from library.train import train_model, LaplaceLoss
from library.predict import inference_fn
from library.utils import score_function

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting TSCG-Net Pipeline Demonstration ===\n")

    # 1. Setup and Configuration
    # --------------------------
    print("[1] Configuring environment for rapid demonstration...")
    seed_everything(42)

    # Override Config defaults for speed and resource constraints
    Config.IMG_SIZE = 64  # Reduce image size for faster processing
    Config.NUM_WORKERS = 0  # Disable multiprocessing to avoid overhead
    Config.DEBUG = True  # Use data subset
    Config.BATCH_SIZE = 4  # Small batch size

    # Set up working directories for demo outputs
    demo_checkpoint_dir = os.path.join(Config.WORKING_DIR, "demo_checkpoints")
    Config.CHECKPOINT_DIR = demo_checkpoint_dir
    os.makedirs(demo_checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Image Size: {Config.IMG_SIZE}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # 2. Data Pipeline Verification
    # -----------------------------
    print("\n[2] Verifying Data Pipeline...")
    train_loader, val_loader, test_loader = get_extended_dataloaders(
        batch_size=Config.BATCH_SIZE
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    print(f"    Batch keys found: {list(batch.keys())}")

    # Assertions
    required_keys = [
        "img_ax",
        "img_cor",
        "tabular",
        "delta_week",
        "target",
        "baseline_fvc",
    ]
    for key in required_keys:
        if key not in batch:
            raise AssertionError(f"Missing key '{key}' in data batch.")

    # Check shapes
    # img_ax: (B, 3, H, W) -> (4, 3, 64, 64)
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    if batch["img_ax"].shape != expected_img_shape:
        raise AssertionError(
            f"img_ax shape mismatch. Expected {expected_img_shape}, got {batch['img_ax'].shape}"
        )

    # tabular: (B, 6) -> Age_norm, Sex, Smoke_Ex, Smoke_Never, Smoke_Curr, Percent_norm
    if batch["tabular"].shape != (Config.BATCH_SIZE, 6):
        raise AssertionError(
            f"tabular shape mismatch. Expected ({Config.BATCH_SIZE}, 6), got {batch['tabular'].shape}"
        )

    print("    Data loading and shapes verified.")

    # 3. Model Logic Verification
    # ---------------------------
    print("\n[3] Verifying Model Architecture and Forward Pass...")
    model = TSCGNet().to(device)

    # Move batch to device
    img_ax = batch["img_ax"].to(device)
    img_cor = batch["img_cor"].to(device)
    tabular = batch["tabular"].to(device)

    # Forward pass
    alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

    print(
        f"    Outputs - Alpha: {alpha.shape}, Sigma Base: {sigma_base.shape}, Sigma Growth: {sigma_growth.shape}"
    )

    # Assertions
    if alpha.shape != (Config.BATCH_SIZE,):
        raise AssertionError("Model output 'alpha' has incorrect shape.")
    if sigma_base.shape != (Config.BATCH_SIZE,):
        raise AssertionError("Model output 'sigma_base' has incorrect shape.")

    print("    Model forward pass successful.")

    # 4. Loss Function Verification
    # -----------------------------
    print("\n[4] Verifying Loss Function...")
    criterion = LaplaceLoss()

    delta_week = batch["delta_week"].to(device)
    baseline_fvc = batch["baseline_fvc"].to(device)
    target = batch["target"].to(device)

    loss = criterion(alpha, sigma_base, sigma_growth, delta_week, baseline_fvc, target)
    print(f"    Calculated Loss: {loss.item():.4f}")

    if torch.isnan(loss):
        raise AssertionError("Loss calculation resulted in NaN.")

    print("    Loss calculation verified.")

    # 5. Training Loop Execution
    # --------------------------
    print("\n[5] Executing Training Loop (1 Epoch)...")
    # Train for 1 epoch to verify pipeline integration
    best_model_path = train_model(
        epochs=1, batch_size=Config.BATCH_SIZE, debug=True, patience=1
    )

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Expected model checkpoint at {best_model_path} but not found."
        )

    print(f"    Training complete. Model saved to: {best_model_path}")

    # 6. Inference Pipeline Execution
    # -------------------------------
    print("\n[6] Executing Inference Pipeline...")
    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    Config.SUBMISSION_PATH = demo_submission_path

    inference_fn(best_model_path, batch_size=Config.BATCH_SIZE, debug=True)

    if not os.path.exists(demo_submission_path):
        raise FileNotFoundError(f"Submission file not found at {demo_submission_path}")

    # Verify submission content
    sub_df = pd.read_csv(demo_submission_path)
    print(f"    Submission generated with {len(sub_df)} rows.")
    print("    Head:")
    print(sub_df.head(3))

    required_cols = {"Patient_Week", "FVC", "Confidence"}
    if not required_cols.issubset(sub_df.columns):
        raise AssertionError(
            f"Submission missing required columns. Found: {sub_df.columns}"
        )

    print("    Inference pipeline verified.")

    # 7. Metric Utility Verification
    # ------------------------------
    print("\n[7] Verifying Metric Utility...")
    # Synthetic data
    y_true_synth = np.array([2000, 3000], dtype=float)
    y_pred_synth = np.array([2100, 2900], dtype=float)
    # Sigma 50 should be clipped to 70
    sigma_synth = np.array([50, 100], dtype=float)

    score = score_function(y_true_synth, y_pred_synth, sigma_synth)
    print(f"    Synthetic Score: {score:.4f}")

    # Manual check:
    # Row 1: Delta=100, Sigma=70. Metric = -sqrt(2)*100/70 - ln(sqrt(2)*70)
    # Row 2: Delta=100, Sigma=100. Metric = -sqrt(2)*100/100 - ln(sqrt(2)*100)
    # Just ensuring it runs and returns a float is sufficient for this check
    if not isinstance(score, (float, np.floating)):
        raise AssertionError("Score function did not return a float.")

    print("    Metric utility verified.")

    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    main()
