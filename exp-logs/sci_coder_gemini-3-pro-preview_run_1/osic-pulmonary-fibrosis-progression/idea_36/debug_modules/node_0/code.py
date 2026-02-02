import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Import library components
from library.config import Config
from library.utils import (
    seed_everything,
    LaplaceLogLikelihoodLoss,
    compute_metric_score,
)
from library.data import get_dataloaders, LungDataset
from library.model import SCVRNet
from library.train import train_one_epoch, validate
from library.predict import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def print_section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def main():
    print_section("1. CONFIGURATION & SETUP")

    # Override Config for fast demonstration
    print("Modifying Config for DEBUG mode...")
    Config.DEBUG = True
    Config.DEBUG_DATA_LIMIT = 20  # Use only 20 patients for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_FILE = "./working/demo_submission.csv"

    # Ensure clean state
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)

    # Setup environment
    Config.setup()
    seed_everything(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    print_section("2. DATA PIPELINE VERIFICATION")

    # Initialize DataLoaders
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))

    img_ax = batch["img_ax"]
    img_cor = batch["img_cor"]
    tabular = batch["tabular"]
    target_fvc = batch["target_fvc"]

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Axial Image Shape: {img_ax.shape} (Expected: [B, 3, 240, 240])")
    print(f"Coronal Image Shape: {img_cor.shape} (Expected: [B, 3, 240, 240])")
    print(f"Tabular Features Shape: {tabular.shape} (Expected: [B, 4])")
    print(f"Target FVC Shape: {target_fvc.shape}")

    # Assertions
    assert img_ax.shape == (
        Config.BATCH_SIZE,
        3,
        240,
        240,
    ), "Incorrect Axial Image Shape"
    assert img_cor.shape == (
        Config.BATCH_SIZE,
        3,
        240,
        240,
    ), "Incorrect Coronal Image Shape"
    assert tabular.shape == (Config.BATCH_SIZE, 4), "Incorrect Tabular Shape"
    assert not torch.isnan(img_ax).any(), "NaNs found in image data"

    print("-> Data Pipeline verified successfully.")

    print_section("3. MODEL ARCHITECTURE CHECK")

    # Initialize Model
    model = SCVRNet().to(device)
    print("SCVRNet instantiated.")

    # Move batch to device
    img_ax = img_ax.to(device)
    img_cor = img_cor.to(device)
    tabular = tabular.to(device)

    # Forward Pass
    print("Performing forward pass...")
    outputs = model(img_ax, img_cor, tabular)

    print(f"Output Shape: {outputs.shape} (Expected: [B, 3])")

    # Assertions
    assert outputs.shape == (Config.BATCH_SIZE, 3), "Model output shape mismatch"
    # Check for NaNs
    if torch.isnan(outputs).any():
        raise ValueError("Model produced NaNs in forward pass")

    print("-> Model Architecture verified successfully.")

    print_section("4. METRIC & LOSS VALIDATION")

    # Test Loss Function
    criterion = LaplaceLogLikelihoodLoss(for_training=True)

    # Create dummy predictions
    # FVC Pred = 2500, True = 2600, Delta = 100
    # Sigma = 50 (should be clipped to 70)
    fvc_pred = torch.tensor([2500.0], device=device)
    fvc_true = torch.tensor([2600.0], device=device)
    sigma_pred = torch.tensor([50.0], device=device)

    loss = criterion(fvc_pred, fvc_true, sigma_pred)

    # Manual Calculation
    # Sigma clipped = 70
    # Delta = 100
    # Metric = - (sqrt(2)*100/70) - ln(sqrt(2)*70)
    # Loss = -Metric = (1.4142 * 100 / 70) + ln(1.4142 * 70)
    # Loss ~= 2.020 + ln(98.99) ~= 2.020 + 4.595 ~= 6.615

    print(f"Computed Loss: {loss.item():.4f}")

    # Check metric score function (should be negative)
    metric = compute_metric_score(fvc_pred, fvc_true, sigma_pred)
    print(f"Computed Metric Score: {metric:.4f}")

    assert loss.item() > 0, "Loss should be positive"
    assert metric < 0, "Metric should be negative"
    assert abs(loss.item() + metric) < 1e-5, "Loss should be -Metric (approximately)"

    print("-> Metric and Loss logic verified.")

    print_section("5. TRAINING LOOP DEMONSTRATION")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("Running training for 1 epoch (Debug subset)...")
    avg_loss = train_one_epoch(
        epoch=1,
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    print(f"Epoch 1 Training Loss: {avg_loss:.4f}")

    # Validate
    val_score = validate(model, val_loader, device)
    print(f"Validation Metric Score: {val_score:.4f}")

    # Save weights for inference step
    demo_model_path = "./working/demo_model.pth"
    torch.save(model.state_dict(), demo_model_path)
    print(f"Model weights saved to {demo_model_path}")

    print("-> Training loop executed successfully.")

    print_section("6. INFERENCE DEMONSTRATION")

    print("Running inference on Test set...")

    # Run inference using the model we just trained/saved
    df_sub = run_inference(
        model_path=demo_model_path,
        output_path=Config.SUBMISSION_FILE,
        device_name=str(device),
    )

    # Verify Submission File
    print(f"\nVerifying submission file: {Config.SUBMISSION_FILE}")
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError("Submission file was not created.")

    df_loaded = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Loaded Submission Shape: {df_loaded.shape}")
    print(f"Columns: {list(df_loaded.columns)}")
    print(df_loaded.head(3))

    # Assertions
    required_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in required_cols:
        assert col in df_loaded.columns, f"Missing column: {col}"

    # Check Confidence clipping in output
    min_conf = df_loaded["Confidence"].min()
    print(f"Minimum Confidence in submission: {min_conf}")
    assert min_conf >= 70.0, "Confidence values found below 70.0 in submission"

    print("-> Inference pipeline verified successfully.")

    print_section("DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    main()
