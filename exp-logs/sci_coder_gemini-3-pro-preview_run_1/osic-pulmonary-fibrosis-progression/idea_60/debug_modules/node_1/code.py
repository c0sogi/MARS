import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.data import LungDataset, get_transforms
from library.model import DCSLNet
from library.train import LaplaceLoss, train_model
from library.utils import seed_everything, load_checkpoint

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=" * 50)
    print("PULMONARY FIBROSIS PROGRESSION - PIPELINE DEMO")
    print("=" * 50)

    # 1. Configuration Overrides for Speed
    print("\n[1] Configuring environment for rapid demonstration...")
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small script
    Config.DEBUG = True  # Use subset of data

    # Ensure reproducible results
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Data Pipeline Verification
    print("\n[2] Verifying Data Pipeline (LungDataset)...")

    # Initialize dataset
    # We use the training CSV and transforms
    dataset = LungDataset(
        csv_path=Config.TRAIN_CSV, mode="train", transform=get_transforms("train")
    )

    # Manually truncate dataset for the unit test to ensure we don't process too many DICOMs
    # (train_model handles this internally when debug=True, but we do it here for the standalone check)
    dataset.df = dataset.df.iloc[:10]

    print(f"Dataset initialized with {len(dataset)} samples (truncated).")

    # Fetch one sample
    sample = dataset[0]

    # Verify keys
    expected_keys = [
        "patient_id",
        "img_axial",
        "img_coronal",
        "tabular",
        "meta_dt",
        "meta_base_fvc",
        "target",
    ]
    for key in expected_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    # Verify Shapes
    # Image: (3, 224, 224)
    assert sample["img_axial"].shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Axial Image Shape: {sample['img_axial'].shape}"
    assert sample["img_coronal"].shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Coronal Image Shape: {sample['img_coronal'].shape}"

    # Tabular: (6,) -> [Age_norm, Sex_enc, Smoke_0, Smoke_1, Smoke_2, Percent_norm]
    assert sample["tabular"].shape == (
        6,
    ), f"Incorrect Tabular Shape: {sample['tabular'].shape}"

    print("Data loading and preprocessing verification passed.")

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture (DCSLNet)...")

    model = DCSLNet().to(device)
    model.eval()

    # Prepare batch (Add batch dimension)
    img_axial = sample["img_axial"].unsqueeze(0).to(device)
    img_coronal = sample["img_coronal"].unsqueeze(0).to(device)
    tabular = sample["tabular"].unsqueeze(0).to(device)

    print(
        f"Input Shapes -> Axial: {img_axial.shape}, Coronal: {img_coronal.shape}, Tabular: {tabular.shape}"
    )

    with torch.no_grad():
        # Output: [alpha, sigma_base, sigma_growth]
        preds = model(img_axial, img_coronal, tabular)

    print(f"Output Shape: {preds.shape}")

    # Verify Output Shape (Batch_Size, 3)
    assert preds.shape == (1, 3), f"Expected output shape (1, 3), got {preds.shape}"

    # Verify Sigma Positivity (Softplus is applied in model)
    sigma_base = preds[0, 1].item()
    sigma_growth = preds[0, 2].item()
    assert sigma_base > 0, "Sigma Base must be positive"
    assert sigma_growth > 0, "Sigma Growth must be positive"

    print("Model forward pass verification passed.")

    # 4. Loss Function Verification
    print("\n[4] Verifying Loss Function (LaplaceLoss)...")

    criterion = LaplaceLoss()

    # Dummy Data
    # True FVC
    target_fvc = torch.tensor([2500.0], device=device)

    # Predicted components from model output
    alpha = preds[:, 0]
    sigma_base = preds[:, 1]
    sigma_growth = preds[:, 2]

    # Metadata for reconstruction
    meta_dt = torch.tensor([10.0], device=device)  # 10 weeks later
    meta_base_fvc = torch.tensor([2600.0], device=device)

    # Reconstruct Prediction
    fvc_pred = meta_base_fvc + alpha * meta_dt
    sigma_pred = sigma_base + sigma_growth * torch.abs(meta_dt)

    # Calculate Loss
    loss = criterion(target_fvc, fvc_pred, sigma_pred)

    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    print("Loss calculation verification passed.")

    # 5. Full Training Loop Demonstration
    print("\n[5] Running Training Loop (Debug Mode)...")

    # This runs the full training pipeline defined in library/train.py
    # debug=True truncates data to 50 samples
    # epochs=2 ensures we test the loop transition
    best_score = train_model(debug=True, epochs=Config.EPOCHS)

    print(f"Training completed. Best Validation Score: {best_score:.4f}")
    assert isinstance(best_score, float), "Best score is not a float"

    # 6. Inference Demonstration
    print("\n[6] Demonstrating Inference with Trained Model...")

    # Load the best model saved during training
    checkpoint_path = (
        "best_model.pth"  # train_model saves to checkpoints/best_model.pth
    )

    # Re-initialize model
    inference_model = DCSLNet().to(device)

    # Load weights
    start_epoch, score = load_checkpoint(
        inference_model, filename=checkpoint_path, device=device
    )
    print(f"Loaded checkpoint from epoch {start_epoch} with score {score:.4f}")

    inference_model.eval()

    # Perform inference on the sample data used earlier
    with torch.no_grad():
        preds = inference_model(img_axial, img_coronal, tabular)

        alpha = preds[:, 0]
        sigma_base = preds[:, 1]
        sigma_growth = preds[:, 2]

        # Predict for a specific week (e.g., dt = 12 weeks)
        dt_input = torch.tensor([12.0], device=device)
        base_fvc_input = torch.tensor([2600.0], device=device)

        final_fvc = base_fvc_input + alpha * dt_input
        final_sigma = sigma_base + sigma_growth * torch.abs(dt_input)

        # Clip confidence as per metric requirement for submission
        final_sigma = torch.clamp(final_sigma, min=70.0)

    print("\n--- Prediction Example ---")
    print(f"Patient Baseline FVC: {base_fvc_input.item():.1f} ml")
    print(f"Time Delta: {dt_input.item()} weeks")
    print(f"Predicted FVC: {final_fvc.item():.2f} ml")
    print(f"Confidence (Sigma): {final_sigma.item():.2f} ml")

    print("\n" + "=" * 50)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
