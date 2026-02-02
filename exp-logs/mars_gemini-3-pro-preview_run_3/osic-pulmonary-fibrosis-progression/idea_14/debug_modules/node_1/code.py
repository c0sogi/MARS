import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import LungDataset, get_dataloaders, get_test_dataloader
from library.model import DDSRNet, predict, loss_fn
from library.train import run_training


def run_demo():
    # 1. Setup and Configuration Overrides for Speed
    print(">>> Setting up demonstration configuration...")
    seed_everything(42)

    # Modify Config for a fast run
    Config.DEBUG = True
    Config.DEBUG_SIZE = 20  # Use only 20 samples
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid overhead for small data
    Config.WORKING_DIR = "./working/demo_task_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Re-run setup to create new directories
    Config.setup()

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # 2. Verify Data Pipeline
    print("\n>>> Verifying Data Pipeline...")

    # Test Dataset instantiation
    train_ds = LungDataset(mode="train")
    print(f"Train Dataset Length (Debug): {len(train_ds)}")

    # Test __getitem__
    sample = train_ds[0]
    print("Sample keys:", sample.keys())

    # Assertions for shapes
    # Image: [3, 384, 384]
    assert sample["image"].shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {sample['image'].shape}"
    # Tabular: [5] -> [BaseFVC_norm, Age_norm, Sex, Smoking, RelWeek_scaled]
    assert sample["tabular"].shape == (
        5,
    ), f"Incorrect tabular shape: {sample['tabular'].shape}"
    # Target: Scalar tensor
    assert isinstance(sample["target"], torch.Tensor), "Target is not a tensor"

    # Test DataLoaders
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=0
    )
    batch = next(iter(train_loader))
    print(f"Batch Image Shape: {batch['image'].shape}")
    print(f"Batch Tabular Shape: {batch['tabular'].shape}")

    assert batch["image"].shape[0] == Config.BATCH_SIZE
    assert batch["tabular"].shape[0] == Config.BATCH_SIZE

    # 3. Verify Model Architecture
    print("\n>>> Verifying Model Architecture...")
    model = DDSRNet().to(Config.DEVICE)

    # Forward pass with the batch
    images = batch["image"].to(Config.DEVICE)
    tabular = batch["tabular"].to(Config.DEVICE)
    targets = batch["target"].to(Config.DEVICE)

    mu, sigma = model(images, tabular)

    print(f"Model Output Shapes - Mu: {mu.shape}, Sigma: {sigma.shape}")

    # Assertions for model output
    assert mu.shape == (Config.BATCH_SIZE, 1), "Incorrect Mu shape"
    assert sigma.shape == (Config.BATCH_SIZE, 1), "Incorrect Sigma shape"

    # Verify Loss Function
    loss = loss_fn(mu, sigma, targets)
    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"

    # 4. Verify Metric Logic
    print("\n>>> Verifying Metric Logic...")
    # Test Case 1: Perfect prediction
    # FVC_true=2000, FVC_pred=2000, Sigma=70 (clipped min)
    # Delta = 0
    # Metric = - (sqrt(2)*0)/70 - ln(sqrt(2)*70) = -ln(98.99) approx -4.595
    val_perfect = laplace_log_likelihood_metric(
        np.array([2000]), np.array([2000]), np.array([10])
    )
    expected_perfect = -np.log(np.sqrt(2) * 70)
    assert np.isclose(
        val_perfect, expected_perfect, atol=1e-4
    ), f"Metric logic failed for perfect prediction. Got {val_perfect}, expected {expected_perfect}"

    # Test Case 2: Large Error (clipped at 1000)
    # FVC_true=2000, FVC_pred=4000 (diff 2000 -> clipped to 1000), Sigma=100
    # Metric = - (sqrt(2)*1000)/100 - ln(sqrt(2)*100)
    #        = - 14.142 - ln(141.42) = -14.142 - 4.95 = -19.09
    val_bad = laplace_log_likelihood_metric(
        np.array([2000]), np.array([4000]), np.array([100])
    )
    expected_bad = -(np.sqrt(2) * 1000) / 100 - np.log(np.sqrt(2) * 100)
    assert np.isclose(
        val_bad, expected_bad, atol=1e-4
    ), f"Metric logic failed for large error. Got {val_bad}, expected {expected_bad}"

    print("Metric logic verified.")

    # 5. Execute Training Loop
    print("\n>>> Executing Training Loop (Integration Test)...")
    best_metric = run_training(patience=1)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Best model checkpoint was not created."
    print(f"Training complete. Best Metric: {best_metric}")

    # 6. Execute Inference
    print("\n>>> Executing Inference...")
    test_loader = get_test_dataloader(batch_size=Config.BATCH_SIZE, num_workers=0)
    submission_df = predict(test_loader)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."
    assert not submission_df.empty, "Submission DataFrame is empty."

    required_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in required_cols:
        assert col in submission_df.columns, f"Missing column in submission: {col}"

    # Check if Confidence is clipped correctly (>= 70)
    min_conf = submission_df["Confidence"].min()
    assert min_conf >= 70, f"Confidence values below 70 found: {min_conf}"

    print("\n>>> Demo execution completed successfully.")


if __name__ == "__main__":
    run_demo()
