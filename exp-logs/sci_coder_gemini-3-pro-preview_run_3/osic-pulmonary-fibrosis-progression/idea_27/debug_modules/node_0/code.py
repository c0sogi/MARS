import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import from the provided library
from library.config import Config
from library.utils import (
    seed_everything,
    LaplaceLogLikelihood,
    calculate_competition_metric,
    TargetScaler,
)
from library.data import get_dataloaders, CTPreprocessor
from library.model import GMARNet
from library.train import Trainer
from library.inference import generate_submission


def run_demo():
    print("=== Starting GMAR-Net Pipeline Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("[1/6] Configuring environment for fast demonstration...")

    # Modify Config for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DEBUG = True  # Use subset of data (100 train, 50 val)

    # Redirect working directories to avoid clutter
    Config.WORKING_DIR = "./working/demo"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create new directories
    for d in [
        Config.WORKING_DIR,
        Config.CACHE_DIR,
        Config.CHECKPOINT_DIR,
        Config.SUBMISSION_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    seed_everything(Config.SEED)
    print("Configuration updated. Output dir: ./working/demo")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n[2/6] Verifying utility functions...")

    # Test TargetScaler
    scaler = TargetScaler()
    data = np.array([1000, 2000, 3000])
    scaler.fit(data)
    transformed = scaler.transform(data)
    inverse = scaler.inverse_transform(transformed)

    # Check reconstruction
    assert np.allclose(data, inverse), "TargetScaler inverse transform failed"

    # Check Sigma inverse (should only scale, not shift)
    sigma_raw = np.array([1.0])
    sigma_inv = scaler.inverse_transform_sigma(sigma_raw)
    expected_sigma = 1.0 * np.std(data)
    assert np.isclose(sigma_inv, expected_sigma), "TargetScaler sigma inverse failed"

    # Test Competition Metric
    # True: 2000, Pred: 2000, Sigma: 100 -> Delta=0, Metric = -ln(sqrt(2)*100)
    # -ln(141.42) approx -4.95
    metric = calculate_competition_metric([2000], [2000], [100])
    expected_metric = -np.log(np.sqrt(2) * 100)
    assert np.isclose(
        metric, expected_metric
    ), f"Metric calculation mismatch: {metric} vs {expected_metric}"

    print("Utilities verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Verify Data Pipeline
    # -------------------------------------------------------------------------
    print("\n[3/6] Verifying Data Pipeline...")

    # Initialize loaders
    train_loader, val_loader, scalers = get_dataloaders(debug=True)

    # Fetch one batch
    batch = next(iter(train_loader))

    images = batch["image"]
    clinical = batch["clinical"]
    targets = batch["target"]

    # Check Shapes
    # Image: (Batch, 3, 260, 260) - 3 slices treated as channels
    assert images.dim() == 4, "Image tensor has incorrect dimensions"
    assert images.shape[1] == 3, "Image tensor should have 3 channels (slices)"
    assert images.shape[2] == Config.IMG_SIZE, "Image height mismatch"

    # Clinical: (Batch, 5) -> [Baseline_FVC, Time, Age, Sex, Smoking]
    assert clinical.dim() == 2, "Clinical tensor has incorrect dimensions"
    assert clinical.shape[1] == 5, "Clinical feature count mismatch"

    print(f"Batch loaded. Images: {images.shape}, Clinical: {clinical.shape}")

    # -------------------------------------------------------------------------
    # 4. Verify Model & Loss
    # -------------------------------------------------------------------------
    print("\n[4/6] Verifying Model and Loss...")

    device = torch.device("cpu")  # Use CPU for simple verification
    model = GMARNet().to(device)
    model.eval()

    with torch.no_grad():
        preds = model(images.to(device), clinical.to(device))

    # Check Output Shape: (Batch, 2) -> [FVC, Confidence]
    assert preds.shape == (Config.BATCH_SIZE, 2), "Model output shape mismatch"

    # Check Confidence Positivity (Sigma)
    sigma_preds = preds[:, 1]
    assert torch.all(sigma_preds > 0), "Model predicted non-positive confidence values"

    # Check Loss Function
    criterion = LaplaceLogLikelihood()
    loss = criterion(preds, targets.to(device))
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() != 0, "Loss is zero (unlikely)"

    print(f"Model forward pass successful. Loss: {loss.item():.4f}")

    # -------------------------------------------------------------------------
    # 5. Run Training Loop
    # -------------------------------------------------------------------------
    print("\n[5/6] Running Training Loop (1 Epoch)...")

    # Initialize Trainer
    # Note: Trainer re-initializes loaders, but uses the Config we patched
    trainer = Trainer(debug=True)

    # Run training
    trainer.train()

    # Verify checkpoint creation
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not created"
    print("Training complete. Checkpoint verified.")

    # -------------------------------------------------------------------------
    # 6. Run Inference
    # -------------------------------------------------------------------------
    print("\n[6/6] Running Inference...")

    generate_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Rows: {len(sub_df)}")

    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Found: {sub_df.columns}"

    # Basic sanity check on values
    assert sub_df["Confidence"].min() >= 70, "Confidence values were not clipped to 70"
    assert not sub_df.isnull().values.any(), "Submission contains NaN values"

    print("Inference verified successfully.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
