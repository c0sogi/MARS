import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, score
from library.data import get_dataloaders, OSICDataset
from library.model import MPVERNet
from library.train import Trainer, LaplaceLogLikelihoodLoss


def run_demo():
    print("=== Starting Lung Function Decline Prediction Demo ===")

    # ------------------------------------------------------------------
    # 1. Configuration Setup
    # ------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Define working paths
    DEMO_DIR = os.path.join(Config.WORKING_DIR, "demo_run")
    CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")

    # Override Config for speed and file permission safety
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = CACHE_DIR
    Config.CHECKPOINT_DIR = CHECKPOINT_DIR
    Config.SUBMISSION_DIR = DEMO_DIR

    # Reduce hyperparameters for demo speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.LR = 1e-3
    Config.PATIENCE = 2

    # Setup directories
    Config.setup()
    seed_everything(Config.SEED)
    print("    Configuration updated and directories created.")

    # ------------------------------------------------------------------
    # 2. Data Preparation (Subsetting)
    # ------------------------------------------------------------------
    print("\n[2] Preparing data subsets...")

    # Load original metadata
    train_meta = pd.read_csv(Config.TRAIN_CSV)
    val_meta = pd.read_csv(Config.VAL_CSV)

    # Create small subsets (e.g., 20 samples)
    train_subset = train_meta.head(20).copy()
    val_subset = val_meta.head(10).copy()

    # Save subsets to working directory
    subset_train_path = os.path.join(DEMO_DIR, "train_subset.csv")
    subset_val_path = os.path.join(DEMO_DIR, "val_subset.csv")

    train_subset.to_csv(subset_train_path, index=False)
    val_subset.to_csv(subset_val_path, index=False)

    # Point Config to these new files
    Config.TRAIN_CSV = subset_train_path
    Config.VAL_CSV = subset_val_path
    print(f"    Created subsets: Train={len(train_subset)}, Val={len(val_subset)}")

    # ------------------------------------------------------------------
    # 3. Metric Verification
    # ------------------------------------------------------------------
    print("\n[3] Verifying Metric Logic...")

    # Manual calculation check
    # Scenario: True=2000, Pred=2100, Sigma=50
    # Sigma clipped = max(50, 70) = 70
    # Delta = min(|2000 - 2100|, 1000) = 100
    # Metric = - (sqrt(2) * 100) / 70 - ln(sqrt(2) * 70)
    #        = - (1.41421356 * 100) / 70 - ln(1.41421356 * 70)
    #        = - 2.0203 - ln(98.9949)
    #        = - 2.0203 - 4.5950
    #        = - 6.6153

    y_true = np.array([2000])
    y_pred = np.array([2100])
    sigma = np.array([50])

    calc_score = score(y_true, y_pred, sigma)
    expected_score = -6.6153

    print(f"    Calculated Score: {calc_score:.4f}")

    # Allow small float precision difference
    assert (
        np.abs(calc_score - expected_score) < 1e-3
    ), f"Metric verification failed! Expected {expected_score}, got {calc_score}"
    print("    Metric verification passed.")

    # ------------------------------------------------------------------
    # 4. Data Loading & Processing
    # ------------------------------------------------------------------
    print("\n[4] Initializing DataLoaders...")

    train_loader, val_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE, val_batch_size=Config.BATCH_SIZE
    )

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))

    # Verify keys
    required_keys = [
        "image_axial",
        "image_coronal",
        "tabular_norm",
        "tabular_raw",
        "time_delta",
        "target",
    ]
    for key in required_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Verify shapes
    # Images: (B, 3, 224, 224)
    img_shape = batch["image_axial"].shape
    assert img_shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {img_shape}"

    # Tabular Norm: (B, 6)
    tab_norm_shape = batch["tabular_norm"].shape
    assert tab_norm_shape == (
        Config.BATCH_SIZE,
        6,
    ), f"Incorrect tabular_norm shape: {tab_norm_shape}"

    print("    DataLoader verified. Batch shapes are correct.")

    # ------------------------------------------------------------------
    # 5. Model Initialization & Forward Pass
    # ------------------------------------------------------------------
    print("\n[5] Initializing MPVERNet Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MPVERNet().to(device)

    # Move batch to device
    img_ax = batch["image_axial"].to(device)
    img_cor = batch["image_coronal"].to(device)
    tab_norm = batch["tabular_norm"].to(device)
    tab_raw = batch["tabular_raw"].to(device)
    time_delta = batch["time_delta"].to(device)

    print("    Running forward pass...")
    pred_fvc, pred_sigma = model(img_ax, img_cor, tab_norm, tab_raw, time_delta)

    # Verify output shapes: (B, 1)
    assert pred_fvc.shape == (Config.BATCH_SIZE, 1), f"Bad FVC shape: {pred_fvc.shape}"
    assert pred_sigma.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Bad Sigma shape: {pred_sigma.shape}"

    # Verify constraints (Sigma must be positive due to softplus)
    assert (pred_sigma >= 0).all(), "Negative confidence values detected!"

    print(f"    Forward pass successful. Pred FVC Mean: {pred_fvc.mean().item():.2f}")

    # ------------------------------------------------------------------
    # 6. Loss Function Check
    # ------------------------------------------------------------------
    print("\n[6] Verifying Loss Function...")

    criterion = LaplaceLogLikelihoodLoss()
    target = batch["target"].to(device)

    loss = criterion(pred_fvc, pred_sigma, target)

    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.dim() == 0, "Loss should be a scalar."

    print(f"    Loss calculated successfully: {loss.item():.4f}")

    # ------------------------------------------------------------------
    # 7. Training Loop Execution
    # ------------------------------------------------------------------
    print("\n[7] Executing Training Loop (2 Epochs)...")

    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit()

    # Check if model checkpoint was saved
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"    Training complete. Best model saved at: {best_model_path}")
    else:
        # If validation score didn't improve (unlikely with random init), check logic
        print("    Training complete (No improvement in 2 epochs or save failed).")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
