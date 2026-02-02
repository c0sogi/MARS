import os
import torch
import pandas as pd
import numpy as np
import sys

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss
from library.data import get_dataloaders
from library.model import PGANet
from library.engine import run_training, generate_submission


def run_demo():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Overrides for Speed
    # We modify the Config class attributes directly to create a fast execution environment
    print("\n[1] Configuring environment for demo...")
    Config.setup()
    seed_everything(Config.SEED)

    # Override Config for speed
    Config.N_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SIZE = 12  # Small subset of patients
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure we use a separate directory for demo checkpoints to avoid overwriting production work
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "demo_checkpoints")
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading and Validation
    print("\n[2] Loading and processing data (Debug Mode)...")
    # This will trigger image caching for the subset of patients
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, debug=True
    )

    # Validate DataLoader output
    print("Validating DataLoader batch structure...")
    batch = next(iter(train_loader))

    required_keys = ["image_axial", "image_coronal", "tabular", "meta", "target"]
    for key in required_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Validate Shapes
    # Images: (B, 3, 224, 224)
    assert batch["image_axial"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    assert batch["image_coronal"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    # Tabular: (B, 4) -> Age, Percent, Sex, Smoking
    assert batch["tabular"].shape == (Config.BATCH_SIZE, 4)
    # Meta: (B, 2) -> Baseline_FVC, Delta_Week
    assert batch["meta"].shape == (Config.BATCH_SIZE, 2)
    # Target: (B, 1) -> FVC
    assert batch["target"].shape == (Config.BATCH_SIZE, 1)

    print("DataLoader validation successful.")

    # 3. Model Instantiation and Forward Pass Verification
    print("\n[3] Instantiating PGA-Net model...")
    model = PGANet()
    model.to(device)

    print("Verifying forward pass...")
    img_ax = batch["image_axial"].to(device)
    img_cor = batch["image_coronal"].to(device)
    tabular = batch["tabular"].to(device)
    meta = batch["meta"].to(device)

    with torch.no_grad():
        pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, meta)

    # Check output shapes: (B,)
    assert pred_fvc.shape == (
        Config.BATCH_SIZE,
    ), f"Expected FVC shape ({Config.BATCH_SIZE},), got {pred_fvc.shape}"
    assert pred_sigma.shape == (
        Config.BATCH_SIZE,
    ), f"Expected Sigma shape ({Config.BATCH_SIZE},), got {pred_sigma.shape}"

    # Check constraints (Sigma must be positive)
    assert (pred_sigma >= 0).all(), "Predicted confidence (sigma) must be non-negative"

    print("Forward pass verification successful.")

    # 4. Loss Function Verification
    print("\n[4] Verifying Laplace Log Likelihood Loss logic...")
    loss_fn = LaplaceLogLikelihoodLoss()

    # Create dummy data
    # Case: True=2000, Pred=2100, Sigma=80
    # Delta = 100
    # Sigma_clipped = max(80, 70) = 80
    # Delta_clipped = min(100, 1000) = 100
    # Metric = - (sqrt(2) * 100 / 80) - ln(sqrt(2) * 80)
    #        = - (1.4142 * 1.25) - ln(113.137)
    #        = - 1.7677 - 4.7286 = -6.496
    # Loss = -Metric = 6.496

    t_fvc = torch.tensor([2000.0], device=device)
    p_fvc = torch.tensor([2100.0], device=device)
    p_sigma = torch.tensor([80.0], device=device)

    calculated_loss = loss_fn(p_fvc, p_sigma, t_fvc).item()

    # Manual calc
    sqrt_2 = np.sqrt(2)
    delta = 100.0
    sigma = 80.0
    expected_metric_val = -(sqrt_2 * delta / sigma) - np.log(sqrt_2 * sigma)
    expected_loss = -expected_metric_val

    assert np.isclose(
        calculated_loss, expected_loss, atol=1e-4
    ), f"Loss mismatch: Got {calculated_loss}, Expected {expected_loss}"

    print(f"Loss verification successful. Value: {calculated_loss:.4f}")

    # 5. Training Loop Execution
    print("\n[5] Running Training Loop (2 Epochs)...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.N_EPOCHS
    )

    best_model_path = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=Config.N_EPOCHS,
        patience=Config.PATIENCE,
    )

    assert os.path.exists(best_model_path), "Best model file was not saved."
    print("Training loop completed successfully.")

    # 6. Inference and Submission
    print("\n[6] Generating Submission...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    generate_submission(model, test_loader, device, output_path=submission_path)

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file not found."

    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(sub_df)} rows.")

    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(sub_df.columns)}"

    # Basic sanity checks on values
    assert not sub_df["FVC"].isnull().any(), "Submission contains NaN FVC values"
    assert (
        not sub_df["Confidence"].isnull().any()
    ), "Submission contains NaN Confidence values"

    print("Submission verification successful.")
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    run_demo()
