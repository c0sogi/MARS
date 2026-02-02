import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, metric_score, InverseScaler
from library.data import get_dataloaders
from library.model import TCDSNet
from library.loss import LaplaceNLLLoss
from library.train import Trainer


def run_demo():
    print("=== Starting TCDS-Net Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid execution...")

    # Override Config defaults to run a tiny, fast experiment
    Config.SEED = 42
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Only use 10 patients
    Config.EPOCHS = 1  # Only 1 epoch
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo
    Config.WORKING_DIR = "./working/demo_task_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Setup directories
    Config.setup()
    seed_everything(Config.SEED)

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # ---------------------------------------------------------
    # 2. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    train_loader, val_loader, test_loader, scalers = get_dataloaders(debug=Config.DEBUG)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches: {len(val_loader)}")
    print(f"    Test Batches: {len(test_loader)}")

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))
    images = batch["image"]
    tabular = batch["tabular"]
    target = batch["target"]

    print(
        f"    Image Shape: {images.shape} (Expected: [{Config.BATCH_SIZE}, 3, {Config.IMG_SIZE}, {Config.IMG_SIZE}])"
    )
    print(f"    Tabular Shape: {tabular.shape} (Expected: [{Config.BATCH_SIZE}, 5])")
    print(f"    Target Shape: {target.shape} (Expected: [{Config.BATCH_SIZE}, 1])")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image shape"
    assert tabular.shape == (Config.BATCH_SIZE, 5), "Incorrect tabular shape"
    assert target.shape == (Config.BATCH_SIZE, 1), "Incorrect target shape"
    assert "fvc_mean" in scalers and "fvc_std" in scalers, "Scalers missing FVC stats"

    print("    Data Pipeline Verified.")

    # ---------------------------------------------------------
    # 3. Model Logic Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = TCDSNet().to(device)

    # Move batch to device
    images = images.to(device)
    tabular = tabular.to(device)

    # Forward pass
    mu, sigma = model(images, tabular)

    print(f"    Output Mu Shape: {mu.shape}")
    print(f"    Output Sigma Shape: {sigma.shape}")

    # Assertions
    assert mu.shape[0] == Config.BATCH_SIZE, "Output batch size mismatch"
    assert sigma.shape[0] == Config.BATCH_SIZE, "Output batch size mismatch"
    # Sigma must be positive (Softplus used in model)
    assert (sigma > 0).all().item(), "Sigma contains non-positive values"

    # Check freezing logic (Backbone should be mostly frozen)
    # The first parameter of the backbone should not require grad
    first_param = next(model.image_encoder.backbone.parameters())
    assert (
        first_param.requires_grad is False
    ), "Backbone freezing logic failed: First layer should be frozen"

    print("    Model Architecture Verified.")

    # ---------------------------------------------------------
    # 4. Loss and Metric Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Loss and Metric...")

    # Loss
    criterion = LaplaceNLLLoss()
    target = target.to(device)
    loss = criterion(mu, sigma, target)

    print(f"    Computed Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # Metric Score Manual Check
    # Scenario: True=2000, Pred=2000, Sigma=100
    # Delta = 0, Sigma_clipped = 100
    # Metric = - (sqrt(2)*0)/100 - ln(sqrt(2)*100) = -ln(141.42) approx -4.95
    y_true = np.array([2000])
    y_pred = np.array([2000])
    sigma_pred = np.array([100])

    score = metric_score(y_true, y_pred, sigma_pred)
    expected_score = -np.log(np.sqrt(2) * 100)

    print(f"    Manual Metric Check: {score:.4f} vs Expected: {expected_score:.4f}")
    assert np.isclose(score, expected_score, atol=1e-3), "Metric calculation incorrect"

    print("    Loss and Metric Verified.")

    # ---------------------------------------------------------
    # 5. Training Loop Execution
    # ---------------------------------------------------------
    print("\n[5] Executing Training Loop (Trainer)...")

    trainer = Trainer(debug=Config.DEBUG)

    # Run fit (1 epoch as configured)
    trainer.fit()

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file not created"

    print("    Training Loop Completed Successfully.")

    # ---------------------------------------------------------
    # 6. Inference Simulation
    # ---------------------------------------------------------
    print("\n[6] Simulating Inference on Test Set...")

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    inverse_scaler = InverseScaler(mean=scalers["fvc_mean"], std=scalers["fvc_std"])

    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            tabs = batch["tabular"].to(device)
            p_weeks = batch["patient_week"]  # List of strings

            mu, sigma = model(imgs, tabs)

            # Inverse transform
            mu_orig, sigma_orig = inverse_scaler(mu, sigma)

            # Collect results
            for i in range(len(p_weeks)):
                predictions.append(
                    {
                        "Patient_Week": p_weeks[i],
                        "FVC": mu_orig[i],
                        "Confidence": sigma_orig[i],
                    }
                )

    # Convert to DataFrame
    sub_df = pd.DataFrame(predictions)
    print(f"    Generated {len(sub_df)} predictions.")
    print("    Sample Predictions:")
    print(sub_df.head(3))

    # Final Format Check
    assert "Patient_Week" in sub_df.columns
    assert "FVC" in sub_df.columns
    assert "Confidence" in sub_df.columns
    assert not sub_df["FVC"].isna().any(), "NaN found in FVC predictions"

    print("    Inference Verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
