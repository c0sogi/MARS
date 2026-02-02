import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.data import get_dataloaders
from library.model import CMTSINModel
from library.losses import MultiTaskLoss
from library.train import run_training
from library.inference import predict_and_submit


def run_demo():
    print("=== Starting Breast Cancer Detection Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Safety
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo execution...")

    # Set specific working directory for this demo to avoid clutter
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Monkey-patch the Config class to run a fast, lightweight version
    Config.WORKING_DIR = demo_working_dir
    Config.SUBMISSION_DIR = demo_working_dir
    Config.MODEL_SAVE_PATH = os.path.join(demo_working_dir, "best_model.pth")
    Config.SUBMISSION_FILE_PATH = os.path.join(demo_working_dir, "submission.csv")

    # Enable Debug mode to use a tiny subset (200 train, 100 val/test)
    Config.DEBUG = True

    # Reduce training parameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple demo

    # Disable pretrained weights to avoid potential network timeouts in isolated envs
    Config.PRETRAINED = False

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Working Dir: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Load dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Train Loader
    batch = next(iter(train_loader))
    images = batch["image"]
    meta = batch["meta"]
    targets = batch["targets"]

    print(f"Train Batch - Image Shape: {images.shape}")
    print(f"Train Batch - Meta Shape: {meta.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    ), f"Incorrect image shape: {images.shape}"
    assert meta.shape == (Config.BATCH_SIZE, 5), f"Incorrect meta shape: {meta.shape}"
    assert "cancer" in targets, "Missing cancer target"
    assert "BIRADS" in targets, "Missing BIRADS target"
    assert "density" in targets, "Missing density target"

    print("Data pipeline verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model and Loss Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture and Loss...")

    device = torch.device("cpu")  # Use CPU for simple logic verification
    model = CMTSINModel().to(device)
    loss_fn = MultiTaskLoss().to(device)

    # Forward pass with the batch fetched earlier
    with torch.no_grad():
        outputs = model(images.to(device), meta.to(device))

    print("Model Output Keys:", outputs.keys())

    # Check outputs
    assert "cancer" in outputs
    assert outputs["cancer"].shape == (
        Config.BATCH_SIZE,
    ), f"Cancer output shape mismatch: {outputs['cancer'].shape}"

    if Config.USE_AUX_TASKS:
        assert "BIRADS" in outputs
        assert "density" in outputs
        assert outputs["BIRADS"].shape == (
            Config.BATCH_SIZE,
            Config.AUX_TASKS["BIRADS"]["num_classes"],
        )
        assert outputs["density"].shape == (
            Config.BATCH_SIZE,
            Config.AUX_TASKS["density"]["num_classes"],
        )

    # Calculate Loss
    # Move targets to device
    targets_device = {k: v.to(device) for k, v in targets.items()}
    loss_dict = loss_fn(outputs, targets_device)

    print("Loss Components:", loss_dict.keys())
    assert "total_loss" in loss_dict
    assert not torch.isnan(loss_dict["total_loss"]), "Loss is NaN"

    print("Model and Loss verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Execution
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop (1 Epoch, Debug Subset)...")

    # This function uses the Config we patched earlier
    run_training()

    # Verify model checkpoint was created
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"

    print("Training execution completed successfully.")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[5] Running Inference and Generating Submission...")

    # This function loads the model we just trained and generates predictions
    predict_and_submit()

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_FILE_PATH
    ), f"Submission file not found at {Config.SUBMISSION_FILE_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE_PATH)
    print(f"Submission File Rows: {len(df_sub)}")
    print(df_sub.head())

    # Basic checks on submission content
    assert "prediction_id" in df_sub.columns
    assert "cancer" in df_sub.columns
    assert len(df_sub) > 0, "Submission file is empty"
    assert (
        df_sub["cancer"].min() >= 0.0 and df_sub["cancer"].max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("Inference and submission verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    run_demo()
