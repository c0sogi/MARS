import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.utils import (
    seed_everything,
    get_device,
    save_submission,
    load_image_with_cache,
    calculate_rmse,
)
from library.wavelet_layers import DWT, IWT
from library.model import WaveCACResUNet, train_one_epoch, validate, predict_tiled
from library.dataset import get_dataloaders
from library.inference import apply_tta


def run_demo():
    # --- Configuration ---
    WORK_DIR = "./working/demo_execution"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    INPUT_DIR = "./input"
    BATCH_SIZE = 4
    PATCH_SIZE = 64  # Smaller patch size for speed in demo
    DEVICE = get_device()

    # Ensure clean working directory
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR, exist_ok=True)

    print(f"--- Starting Demo on {DEVICE} ---")
    seed_everything(42)

    # ==========================================
    # 1. Validate Wavelet Layers (DWT & IWT)
    # ==========================================
    print("\n[1] Validating Wavelet Layers...")

    dwt = DWT().to(DEVICE)
    iwt = IWT().to(DEVICE)

    # Create a dummy input: (Batch=1, Channel=1, Height=32, Width=32)
    # Using simple values to verify reconstruction
    dummy_input = torch.randn(1, 1, 32, 32).to(DEVICE)

    # Forward DWT
    freq_subbands = dwt(dummy_input)
    # Expected shape: (1, 4, 16, 16)
    assert freq_subbands.shape == (
        1,
        4,
        16,
        16,
    ), f"DWT output shape mismatch. Expected (1, 4, 16, 16), got {freq_subbands.shape}"

    # Inverse IWT
    reconstructed = iwt(freq_subbands)
    # Expected shape: (1, 1, 32, 32)
    assert reconstructed.shape == (
        1,
        1,
        32,
        32,
    ), f"IWT output shape mismatch. Expected (1, 1, 32, 32), got {reconstructed.shape}"

    # Check Reconstruction Error (Haar should be near perfect for float32)
    recon_error = torch.mean(torch.abs(dummy_input - reconstructed)).item()
    print(f"    Wavelet Reconstruction Error: {recon_error:.6f}")
    assert recon_error < 1e-5, "Wavelet reconstruction error is too high!"

    print("    Wavelet layers validated successfully.")

    # ==========================================
    # 2. Validate Model Architecture (WaveCACResUNet)
    # ==========================================
    print("\n[2] Validating WaveCACResUNet Model...")

    model = WaveCACResUNet(in_channels=1, base_filters=16).to(
        DEVICE
    )  # Reduced filters for speed

    # Pass dummy input
    output = model(dummy_input)

    # Check output shape (should match input shape for UNet)
    assert (
        output.shape == dummy_input.shape
    ), f"Model output shape mismatch. Expected {dummy_input.shape}, got {output.shape}"

    # Check gradient flow
    target = torch.randn_like(output)
    loss = nn.MSELoss()(output, target)
    loss.backward()

    # Check if a parameter has gradients
    param_with_grad = next(model.parameters())
    assert (
        param_with_grad.grad is not None
    ), "Model parameters have no gradients after backward pass!"

    print("    Model architecture instantiated and forward/backward pass successful.")

    # ==========================================
    # 3. Validate Dataset and Dataloaders
    # ==========================================
    print("\n[3] Validating Data Loading...")

    # We use a very small samples_per_epoch to make the epoch short
    train_loader, val_loader = get_dataloaders(
        data_dir=INPUT_DIR,
        cache_dir=CACHE_DIR,
        batch_size=BATCH_SIZE,
        num_workers=2,
        patch_size=PATCH_SIZE,
        train_samples_per_epoch=1,  # Minimal sampling
        val_samples_per_epoch=1,
    )

    print(f"    Train Loader Length (batches): {len(train_loader)}")
    print(f"    Val Loader Length (batches): {len(val_loader)}")

    # Fetch one batch
    noisy_batch, clean_batch = next(iter(train_loader))

    # Verify shapes
    expected_shape = (BATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE)
    assert (
        noisy_batch.shape == expected_shape
    ), f"Batch shape mismatch. Expected {expected_shape}, got {noisy_batch.shape}"
    assert (
        clean_batch.shape == expected_shape
    ), f"Label shape mismatch. Expected {expected_shape}, got {clean_batch.shape}"

    # Verify value range [0, 1]
    assert (
        noisy_batch.min() >= 0 and noisy_batch.max() <= 1.0
    ), "Noisy image values out of range [0, 1]"
    assert (
        clean_batch.min() >= 0 and clean_batch.max() <= 1.0
    ), "Clean image values out of range [0, 1]"

    print("    Data loading and preprocessing validated.")

    # ==========================================
    # 4. Validate Training Loop
    # ==========================================
    print("\n[4] Validating Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Train for one epoch
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    print(f"    Training finished. Loss: {train_loss:.6f}")
    assert not np.isnan(train_loss), "Training loss is NaN!"

    # Validate
    val_rmse = validate(model, val_loader, DEVICE)
    print(f"    Validation finished. RMSE: {val_rmse:.6f}")
    assert not np.isnan(val_rmse), "Validation RMSE is NaN!"

    # ==========================================
    # 5. Validate Inference and TTA
    # ==========================================
    print("\n[5] Validating Inference and TTA...")

    # Pick a sample image from validation set for inference testing
    # We'll manually load one to simulate the inference pipeline
    val_metadata = pd.read_csv("./metadata/val.csv")
    sample_row = val_metadata.iloc[0]
    sample_img_path = os.path.join(INPUT_DIR, sample_row["feature_path"])

    # Load image
    sample_img = load_image_with_cache(
        sample_img_path, os.path.join(CACHE_DIR, "sample_test.npy")
    )

    # 1. Tiled Prediction
    pred_clean = predict_tiled(model, sample_img, DEVICE, patch_size=PATCH_SIZE)

    assert (
        pred_clean.shape == sample_img.shape
    ), f"Prediction shape mismatch. Expected {sample_img.shape}, got {pred_clean.shape}"

    # 2. Test Time Augmentation (TTA)
    pred_tta = apply_tta(model, sample_img, DEVICE)

    assert (
        pred_tta.shape == sample_img.shape
    ), f"TTA Prediction shape mismatch. Expected {sample_img.shape}, got {pred_tta.shape}"

    print("    Inference functions validated.")

    # ==========================================
    # 6. Validate Submission Generation
    # ==========================================
    print("\n[6] Validating Submission Generation...")

    submission_path = os.path.join(WORK_DIR, "submission_test.csv")

    # Create dummy predictions dictionary
    # Using the sample image ID
    dummy_preds = {sample_row["id"]: pred_tta}

    save_submission(dummy_preds, submission_path)

    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify file content format
    df_sub = pd.read_csv(submission_path)
    print(f"    Submission file created with {len(df_sub)} rows.")

    expected_cols = ["id", "value"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}"

    # Check ID format (image_row_col)
    first_id = df_sub.iloc[0]["id"]
    parts = first_id.split("_")
    assert len(parts) >= 3, f"Submission ID format incorrect. Got {first_id}"

    print("    Submission generation validated.")
    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
