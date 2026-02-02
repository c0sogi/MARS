import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss
from library.data import get_dataloaders
from library.model import BBSLNet
from library.engine import train_fn, eval_fn, inference_fn


def run_demo():
    print("=== Starting BBSL-Net Demo Script ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup for Demo
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid execution...")

    # Override Config for speed and isolation
    Config.debug = True
    Config.epochs = 1
    Config.batch_size = 4
    Config.num_workers = 0  # Avoid multiprocessing overhead in demo
    Config.working_dir = "./working/demo_run"
    Config.cache_dir = os.path.join(Config.working_dir, "cache")
    Config.model_save_path = os.path.join(
        Config.working_dir, "checkpoints", "best_model.pth"
    )
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")

    # Create necessary directories
    os.makedirs(Config.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(Config.model_save_path), exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.seed)
    device = torch.device(Config.device)
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.debug}")
    print(f"    Working Directory: {Config.working_dir}")

    # ---------------------------------------------------------
    # 2. Data Loading Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Get dataloaders
    train_loader, val_loader, sub_loader = get_dataloaders()

    # Fetch one batch from training loader
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty! Check dataset availability.")

    # Unpack batch
    img_ax = batch["img_ax"].to(device)
    img_cor = batch["img_cor"].to(device)
    tabular = batch["tabular"].to(device)
    time_delta = batch["time_delta"].to(device)
    baseline_fvc = batch["baseline_fvc"].to(device)
    targets = batch["target"].to(device)

    # Assertions for shapes
    print("    Checking input shapes...")
    # Image: (Batch, 3, 224, 224)
    expected_img_shape = (Config.batch_size, 3, Config.image_size, Config.image_size)
    assert (
        img_ax.shape == expected_img_shape
    ), f"Axial Image shape mismatch: {img_ax.shape}"
    assert (
        img_cor.shape == expected_img_shape
    ), f"Coronal Image shape mismatch: {img_cor.shape}"

    # Tabular: (Batch, 4) -> Age, Sex, Smoking, Percent
    assert tabular.shape == (
        Config.batch_size,
        4,
    ), f"Tabular shape mismatch: {tabular.shape}"

    # Targets: (Batch,)
    assert targets.shape == (
        Config.batch_size,
    ), f"Target shape mismatch: {targets.shape}"

    print("    Data shapes verified successfully.")

    # ---------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = BBSLNet()
    model.to(device)

    # Forward pass
    preds = model(img_ax, img_cor, tabular, time_delta, baseline_fvc)

    # Check output shape: (Batch, 2) -> [FVC, Confidence]
    assert preds.shape == (
        Config.batch_size,
        2,
    ), f"Prediction shape mismatch: {preds.shape}"

    print("    Forward pass successful.")
    print(f"    Prediction Mean FVC: {preds[:, 0].mean().item():.2f}")
    print(f"    Prediction Mean Sigma: {preds[:, 1].mean().item():.2f}")

    # ---------------------------------------------------------
    # 4. Loss Function Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Loss Function...")

    loss_fn = LaplaceLogLikelihoodLoss()
    loss = loss_fn(preds, targets)

    assert torch.isfinite(loss), "Loss is not finite (NaN or Inf)"
    print(f"    Calculated Loss: {loss.item():.4f}")

    # ---------------------------------------------------------
    # 5. Training Step Verification
    # ---------------------------------------------------------
    print("\n[5] Verifying Training Step...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Run train_fn for one iteration (epoch logic is inside engine, but we call it once)
    # Since train_fn iterates over the whole loader, and we are in debug mode (small data),
    # this will be quick.
    train_loss = train_fn(train_loader, model, optimizer, device, loss_fn)
    print(f"    Train Function executed. Avg Loss: {train_loss:.4f}")

    # Run eval_fn
    val_loss = eval_fn(val_loader, model, device, loss_fn)
    print(f"    Eval Function executed. Avg Loss: {val_loss:.4f}")

    # Save a dummy checkpoint for inference test
    torch.save(model.state_dict(), Config.model_save_path)
    print("    Dummy model checkpoint saved.")

    # ---------------------------------------------------------
    # 6. Inference Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Inference Pipeline...")

    # Run inference
    inference_fn(sub_loader, model, device)

    # Check if submission file exists
    assert os.path.exists(Config.submission_path), "Submission file was not created."

    # Validate submission content
    sub_df = pd.read_csv(Config.submission_path)
    print(f"    Submission file loaded. Rows: {len(sub_df)}")

    required_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in required_cols:
        assert col in sub_df.columns, f"Missing column in submission: {col}"

    # Check values
    assert not sub_df["FVC"].isnull().any(), "NaN found in FVC predictions"
    assert (
        not sub_df["Confidence"].isnull().any()
    ), "NaN found in Confidence predictions"

    # Check metric constraint (Confidence >= 70)
    min_conf = sub_df["Confidence"].min()
    assert (
        min_conf >= 70
    ), f"Confidence clipping failed. Min confidence found: {min_conf}"

    print("    Submission format and constraints verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
