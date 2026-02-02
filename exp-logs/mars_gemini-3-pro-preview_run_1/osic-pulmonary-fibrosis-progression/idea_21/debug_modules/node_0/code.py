import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data import get_dataloaders
from library.model import CG_SDAN, criterion, predict
from library.train import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Patch Config to run a small, fast experiment
    Config.N_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_DATA_SIZE = 10  # Only use 10 patients per split
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Redirect outputs to a demo-specific directory
    Config.IDEA_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.IDEA_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.IDEA_DIR, "checkpoints")
    Config.MODEL_SAVE_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.IDEA_DIR, "submission.csv")

    # Re-run setup to create these new directories
    Config.setup()

    # Set seed
    seed_everything(Config.SEED)
    print("    Configuration patched. Output dir:", Config.IDEA_DIR)

    # ------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # ------------------------------------------------------------------------
    print("\n[2] Initializing DataLoaders (Debug Mode)...")

    # This will trigger preprocessing for the small debug subset
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    # Assertions to verify data integrity
    assert "image_axial" in batch, "Batch missing 'image_axial'"
    assert "image_coronal" in batch, "Batch missing 'image_coronal'"
    assert "tabular" in batch, "Batch missing 'tabular'"
    assert "target" in batch, "Batch missing 'target'"
    assert "meta" in batch, "Batch missing 'meta'"

    # Verify Shapes
    # Image: (B, 3, 224, 224)
    img_shape = batch["image_axial"].shape
    assert img_shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Unexpected image shape: {img_shape}"

    # Tabular: (B, 5) -> [Week, Percent, Age, Sex, Smoke]
    tab_shape = batch["tabular"].shape
    assert tab_shape == (Config.BATCH_SIZE, 5), f"Unexpected tabular shape: {tab_shape}"

    print("    Batch structure verified successfully.")

    # ------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # ------------------------------------------------------------------------
    print("\n[3] Instantiating CG_SDAN Model and running forward pass...")

    device = torch.device(Config.DEVICE)
    model = CG_SDAN().to(device)

    # Move batch to device
    img_ax = batch["image_axial"].to(device)
    img_cor = batch["image_coronal"].to(device)
    tabular = batch["tabular"].to(device)

    # Forward pass
    alpha, s_base, s_growth = model(img_ax, img_cor, tabular)

    # Verify outputs
    # All outputs should be (B, 1)
    assert alpha.shape == (Config.BATCH_SIZE, 1), f"Alpha shape mismatch: {alpha.shape}"
    assert s_base.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Sigma Base shape mismatch: {s_base.shape}"
    assert s_growth.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Sigma Growth shape mismatch: {s_growth.shape}"

    # Verify gradients are tracked
    assert alpha.requires_grad, "Alpha output detached from computation graph"

    print("    Forward pass successful. Output shapes verified.")

    # ------------------------------------------------------------------------
    # 4. Loss Calculation Verification
    # ------------------------------------------------------------------------
    print("\n[4] calculating Loss...")

    targets = batch["target"].to(device)
    m_weeks = batch["meta"]["Weeks"].to(device).view(-1, 1)
    m_base_fvc = batch["meta"]["Baseline_FVC"].to(device).view(-1, 1)
    m_base_week = batch["meta"]["Baseline_Week"].to(device).view(-1, 1)

    loss = criterion(alpha, s_base, s_growth, targets, m_weeks, m_base_fvc, m_base_week)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() != 0.0, "Loss is zero (unexpected)"

    # Verify backward pass
    loss.backward()
    print(f"    Loss calculation successful. Loss Value: {loss.item():.4f}")

    # ------------------------------------------------------------------------
    # 5. Full Training Loop Simulation
    # ------------------------------------------------------------------------
    print("\n[5] Running Training Loop (2 Epochs)...")

    # run_training handles the loop, validation, and saving
    run_training(debug=True)

    # Verify model checkpoint was created
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"    Training complete. Model saved to: {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
        )

    # ------------------------------------------------------------------------
    # 6. Inference & Submission Generation
    # ------------------------------------------------------------------------
    print("\n[6] Generating Predictions on Test Set...")

    # Predict using the test loader
    predict(test_loader)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission loaded. Shape: {sub_df.shape}")
    print(f"    Columns: {list(sub_df.columns)}")

    # Verify Columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in expected_cols:
        assert col in sub_df.columns, f"Missing column in submission: {col}"

    # Verify Confidence Constraint (>= 70)
    min_conf = sub_df["Confidence"].min()
    assert min_conf >= 70, f"Found confidence value < 70: {min_conf}"

    print(f"    Submission valid. Min Confidence: {min_conf}")

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
