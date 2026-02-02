import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import library components
from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import VCDAN, generate_submission, LaplaceLoss
from library.train import run_training, train_one_epoch, evaluate
from library.utils import calculate_metric

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def demo_pipeline():
    print("=== Starting Demonstration Pipeline ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set paths to a temporary demo directory
    DEMO_DIR = "./working/demo_run"
    CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Override Config attributes
    Config.IDEA_DIR = DEMO_DIR
    Config.CACHE_DIR = CACHE_DIR
    Config.MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = SUBMISSION_PATH

    # Reduce dataset size and training duration
    Config.MAX_TRAIN_SAMPLES = 16  # Only use 16 samples for training/validation
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo
    Config.PATIENCE = 2  # Short patience

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated.")

    # ---------------------------------------------------------
    # 2. Data Loading & Verification
    # ---------------------------------------------------------
    print("\n[2] Loading and verifying data...")

    # Load dataloaders (this will trigger image processing for the subset)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # Fetch a single batch to verify shapes and types
    batch = next(iter(train_loader))
    img_ax = batch["img_ax"]
    img_cor = batch["img_cor"]
    tabular = batch["tabular"]
    target = batch["target"]
    patient_weeks = batch["patient_week"]

    # Assertions
    assert img_ax.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Axial Image shape: {img_ax.shape}"
    assert img_cor.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Coronal Image shape: {img_cor.shape}"
    # Tabular dim is 6: Age, Sex, Smoke, Percent, Base_FVC, Rel_Week
    assert tabular.shape == (
        Config.BATCH_SIZE,
        6,
    ), f"Incorrect Tabular shape: {tabular.shape}"
    assert target.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect Target shape: {target.shape}"

    print("Data shapes verified successfully.")

    # ---------------------------------------------------------
    # 3. Model Initialization & Forward Pass Verification
    # ---------------------------------------------------------
    print("\n[3] Initializing model and verifying forward pass...")

    device = torch.device(Config.DEVICE)
    model = VCDAN().to(device)

    # Move batch to device
    img_ax = img_ax.to(device)
    img_cor = img_cor.to(device)
    tabular = tabular.to(device)

    # Forward pass
    fvc_pred, conf_pred = model(img_ax, img_cor, tabular)

    # Verify output shapes
    assert fvc_pred.shape == (Config.BATCH_SIZE,), "FVC prediction shape mismatch"
    assert conf_pred.shape == (
        Config.BATCH_SIZE,
    ), "Confidence prediction shape mismatch"

    # Verify confidence positivity (Softplus output)
    assert (conf_pred > 0).all(), "Confidence values must be positive"

    print(f"Model output verified. FVC Pred: {fvc_pred[:2].detach().cpu().numpy()}")

    # ---------------------------------------------------------
    # 4. Loss & Metric Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Loss and Metric calculation...")

    criterion = LaplaceLoss()
    target = target.to(device)

    # Calculate loss
    loss = criterion(fvc_pred, conf_pred, target)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.dim() == 0, "Loss should be a scalar"

    # Calculate metric
    metric_val = calculate_metric(target, fvc_pred, conf_pred)
    print(f"Calculated Metric: {metric_val:.4f}")

    assert isinstance(metric_val, float), "Metric should be a float"

    print("Loss and Metric logic verified.")

    # ---------------------------------------------------------
    # 5. Full Training Loop Execution
    # ---------------------------------------------------------
    print("\n[5] Executing Training Loop (run_training)...")

    # We use the library function which handles the loop, checkpointing, etc.
    # It uses the Config we patched earlier.
    run_training()

    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Best model checkpoint was not created."

    print("Training loop completed successfully.")

    # ---------------------------------------------------------
    # 6. Inference & Submission Generation
    # ---------------------------------------------------------
    print("\n[6] Generating Submission...")

    # To speed up inference demo, we can slice the test dataset in the loader
    # However, generate_submission re-initializes loaders.
    # Since test set is small enough (18 patients), we let it run normally.
    generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    required_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in sub_df.columns for col in required_cols
    ), f"Submission missing columns. Found: {sub_df.columns}"

    print(f"Submission generated with {len(sub_df)} rows.")
    print(sub_df.head(3))

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    demo_pipeline()
