import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import get_dataloaders
from library.model import PRTNet
from library.train import run_training, LaplaceLogLikelihoodLoss

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_config():
    """
    Overrides Config parameters to run a fast, small-scale demonstration.
    """
    print(">>> Setting up Demo Configuration...")

    # 1. Enable Debug Mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use 20 samples for speed

    # 2. Reduce Training duration
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # 3. Set a specific project name for this demo run
    Config.PROJECT_NAME = "demo_task_execution"

    # 4. Update dependent paths (since they were initialized at import time)
    Config.WORKING_DIR = os.path.join("./working", Config.PROJECT_NAME)
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Set device
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"    Project: {Config.PROJECT_NAME}")
    print(f"    Device: {Config.DEVICE}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Epochs: {Config.NUM_EPOCHS}")


def validate_data_pipeline():
    """
    Demonstrates and validates the data loading pipeline.
    """
    print("\n>>> Validating Data Pipeline...")

    # Load dataloaders
    # This triggers caching of images and preprocessing of tabular data
    train_loader, val_loader, test_loader, stats = get_dataloaders(
        load_cached_data=True
    )

    # Check if loaders are populated
    assert len(train_loader) > 0, "Train loader is empty!"
    assert len(val_loader) > 0, "Val loader is empty!"

    # Fetch a single batch
    batch = next(iter(train_loader))

    # Verify Batch Keys
    expected_keys = {"image", "static", "rel_time", "target", "raw_fvc"}
    assert expected_keys.issubset(
        batch.keys()
    ), f"Batch missing keys. Found: {batch.keys()}"

    # Verify Shapes
    # Image: (B, 3, H, W) -> (4, 3, 256, 256)
    images = batch["image"]
    assert images.dim() == 4, f"Image dim mismatch. Expected 4, got {images.dim()}"
    assert (
        images.shape[1] == 3
    ), f"Image channels mismatch. Expected 3, got {images.shape[1]}"
    assert images.shape[2] == Config.IMG_SIZE, "Image height mismatch"

    # Static features: (B, 4) -> [Baseline_FVC, Age, Sex, Smoking]
    static = batch["static"]
    assert (
        static.shape[1] == 4
    ), f"Static features mismatch. Expected 4, got {static.shape[1]}"

    # Target: (B,)
    target = batch["target"]
    assert target.dim() == 1, "Target should be 1D tensor"

    print("    Data Pipeline Validated. Batch shapes are correct.")
    return batch, stats


def validate_model_architecture(batch):
    """
    Demonstrates model instantiation and validates the forward pass.
    """
    print("\n>>> Validating Model Architecture...")

    device = Config.DEVICE
    model = PRTNet().to(device)
    model.eval()

    # Move batch to device
    images = batch["image"].to(device)
    static = batch["static"].to(device)
    rel_time = batch["rel_time"].to(device)

    # Forward Pass
    with torch.no_grad():
        mu, sigma = model(images, static, rel_time)

    # Check Output Shapes
    batch_size = images.size(0)
    assert mu.shape == (
        batch_size,
    ), f"Mu shape mismatch. Expected ({batch_size},), got {mu.shape}"
    assert sigma.shape == (
        batch_size,
    ), f"Sigma shape mismatch. Expected ({batch_size},), got {sigma.shape}"

    # Check Sigma Positivity (Model uses Softplus + Offset)
    assert torch.all(sigma > 0), "Sigma values must be positive!"

    print(
        f"    Model Forward Pass Successful. Outputs: mu={mu.shape}, sigma={sigma.shape}"
    )

    return model, mu, sigma


def validate_loss_and_metric(mu, sigma, batch, stats):
    """
    Validates the custom loss function and the competition metric.
    """
    print("\n>>> Validating Loss and Metric...")

    device = Config.DEVICE
    target_scaled = batch["target"].to(device)
    raw_fvc = batch["raw_fvc"].to(device)

    # 1. Loss Calculation (on scaled data)
    criterion = LaplaceLogLikelihoodLoss()
    loss = criterion(mu, sigma, target_scaled)

    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() != 0, "Loss is zero (unlikely for random init)"
    print(f"    Calculated Loss: {loss.item():.4f}")

    # 2. Metric Calculation (on absolute data)
    # Inverse transform predictions
    fvc_mean = stats["fvc_mean"]
    fvc_std = stats["fvc_std"]

    mu_abs = mu * fvc_std + fvc_mean
    sigma_abs = sigma * fvc_std

    metric = laplace_log_likelihood_metric(raw_fvc, mu_abs, sigma_abs)

    assert isinstance(metric, float), "Metric should return a float"
    # Metric is negative and higher is better, usually around -6 to -10 for random init
    print(f"    Calculated Metric: {metric:.4f}")


def run_full_training_demo():
    """
    Runs the full training loop using the library's train module.
    """
    print("\n>>> Running Full Training Loop (Demo)...")

    # This function inside library.train handles the loop, validation, and checkpointing
    run_training()

    # Verify Checkpoint Creation
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not created!"
    print(f"    Training Complete. Checkpoint saved at: {Config.BEST_MODEL_PATH}")


def validate_inference():
    """
    Demonstrates how to load the saved model and perform inference on the test set.
    """
    print("\n>>> Validating Inference on Test Set...")

    device = Config.DEVICE

    # Load Data
    _, _, test_loader, stats = get_dataloaders(load_cached_data=True)

    # Load Model
    model = PRTNet().to(device)
    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            static = batch["static"].to(device)
            rel_time = batch["rel_time"].to(device)
            patient_weeks = batch["patient_week"]

            mu_scaled, sigma_scaled = model(images, static, rel_time)

            # Inverse Transform
            mu_abs = mu_scaled.cpu().numpy() * stats["fvc_std"] + stats["fvc_mean"]
            sigma_abs = sigma_scaled.cpu().numpy() * stats["fvc_std"]

            for pw, fvc, conf in zip(patient_weeks, mu_abs, sigma_abs):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    # Create Submission DataFrame
    sub_df = pd.DataFrame(results)
    print(f"    Inference generated {len(sub_df)} predictions.")

    # Check format
    assert "Patient_Week" in sub_df.columns
    assert "FVC" in sub_df.columns
    assert "Confidence" in sub_df.columns

    print("    Inference Validation Successful.")
    print(sub_df.head())


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    # 1. Setup Configuration
    setup_demo_config()

    # 2. Validate Components
    batch_data, data_stats = validate_data_pipeline()
    model, mu_pred, sigma_pred = validate_model_architecture(batch_data)
    validate_loss_and_metric(mu_pred, sigma_pred, batch_data, data_stats)

    # 3. Run Integration Test (Training)
    run_full_training_demo()

    # 4. Run Inference Test
    validate_inference()

    print("\n>>> All demonstrations completed successfully.")
