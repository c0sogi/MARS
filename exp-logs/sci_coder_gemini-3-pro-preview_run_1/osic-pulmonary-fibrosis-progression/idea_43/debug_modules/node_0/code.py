import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.data import get_dataloaders, get_test_loader
from library.model import BSHDAN
from library.utils import LaplaceLogLikelihoodLoss, calculate_metric, seed_everything
from library.train import train_model, generate_submission


def main():
    print("Initializing Demonstration Script...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Modify Config for a fast demonstration run
    print("Configuring for demo run...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SIZE = 12  # Small subset for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    Config.PATIENCE = 1

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Device check
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n--- Verifying Data Loading ---")

    # Get dataloaders in debug mode
    train_loader, val_loader = get_dataloaders(debug=True)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch a single batch
    batch = next(iter(train_loader))

    # Verify keys
    expected_keys = [
        "img_axial",
        "img_coronal",
        "tabular",
        "delta_week",
        "baseline_fvc",
        "target",
        "patient_week",
    ]
    for key in expected_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Verify Shapes
    # Image: (B, 3, 224, 224)
    img_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        batch["img_axial"].shape == img_shape
    ), f"Axial shape mismatch: {batch['img_axial'].shape}"
    assert (
        batch["img_coronal"].shape == img_shape
    ), f"Coronal shape mismatch: {batch['img_coronal'].shape}"

    # Tabular: (B, 7) -> [Percent, Age, Sex*2, Smoke*3]
    assert batch["tabular"].shape == (
        Config.BATCH_SIZE,
        7,
    ), f"Tabular shape mismatch: {batch['tabular'].shape}"

    # Target: (B,)
    assert batch["target"].shape == (
        Config.BATCH_SIZE,
    ), f"Target shape mismatch: {batch['target'].shape}"

    print("Data loading verified successfully.")

    # ==========================================
    # 3. Model & Forward Pass Verification
    # ==========================================
    print("\n--- Verifying Model Architecture ---")

    model = BSHDAN().to(device)

    # Move batch to device
    img_axial = batch["img_axial"].to(device)
    img_coronal = batch["img_coronal"].to(device)
    tabular = batch["tabular"].to(device)
    delta_week = batch["delta_week"].to(device)
    baseline_fvc = batch["baseline_fvc"].to(device)

    # Perform forward pass
    outputs = model(img_axial, img_coronal, tabular, delta_week, baseline_fvc)

    # Verify outputs
    assert "fvc_pred" in outputs, "Model output missing 'fvc_pred'"
    assert "sigma_pred" in outputs, "Model output missing 'sigma_pred'"
    assert "alpha" in outputs, "Model output missing 'alpha'"

    # Check output shapes
    assert outputs["fvc_pred"].shape == (
        Config.BATCH_SIZE,
    ), "Prediction shape mismatch"
    assert outputs["sigma_pred"].shape == (
        Config.BATCH_SIZE,
    ), "Confidence shape mismatch"

    # Check logic: Sigma should be positive (Softplus used in model)
    assert (outputs["sigma_pred"] > 0).all(), "Sigma predictions must be positive"

    print("Model forward pass verified successfully.")

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print("\n--- Verifying Loss Function ---")

    criterion = LaplaceLogLikelihoodLoss()
    target = batch["target"].to(device)

    loss = criterion(outputs["fvc_pred"], outputs["sigma_pred"], target)

    # Check loss is a scalar tensor
    assert isinstance(loss, torch.Tensor), "Loss must be a tensor"
    assert loss.dim() == 0, "Loss must be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"

    print(f"Calculated Loss: {loss.item():.4f}")
    print("Loss function verified successfully.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n--- Running Training Loop (Demo) ---")

    # Clean up any existing checkpoints for a fresh demo
    if os.path.exists(Config.CHECKPOINT_DIR):
        shutil.rmtree(Config.CHECKPOINT_DIR)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Run training
    # This uses the modified Config (1 Epoch, Debug mode)
    trained_model = train_model(debug=True)

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not created."

    print("Training loop completed successfully.")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\n--- Generating Submission ---")

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Generate submission using the trained model
    generate_submission(model=trained_model)

    # Verify submission file
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file was not created."

    df_sub = pd.read_csv(sub_path)
    print(f"Submission shape: {df_sub.shape}")

    # Check columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Found: {list(df_sub.columns)}"

    # Check content (no NaNs)
    assert not df_sub.isnull().values.any(), "Submission contains NaNs"

    print("Submission generation verified successfully.")

    # ==========================================
    # 7. Metric Calculation Utility
    # ==========================================
    print("\n--- Verifying Metric Calculation ---")

    # Create dummy data
    y_true = np.array([2000, 2500, 3000])
    y_pred = np.array([2100, 2400, 2000])  # Errors: 100, 100, 1000
    sigma_pred = np.array([100, 100, 50])  # Note: 50 should be clipped to 70

    score = calculate_metric(y_true, y_pred, sigma_pred)

    # Manual check
    # Item 1: Delta=100, Sigma=100. Metric = -sqrt(2)*100/100 - ln(sqrt(2)*100) = -1.414 - 4.95 = -6.36
    # Item 2: Delta=100, Sigma=100. Metric = -6.36
    # Item 3: Delta=1000, Sigma=70 (clipped). Metric = -sqrt(2)*1000/70 - ln(sqrt(2)*70) = -20.2 - 4.6 = -24.8
    # Mean approx -12.5

    assert isinstance(score, float), "Metric should return a float"
    print(f"Calculated Dummy Score: {score:.4f}")

    print("\nAll demonstrations and verifications passed!")


if __name__ == "__main__":
    main()
