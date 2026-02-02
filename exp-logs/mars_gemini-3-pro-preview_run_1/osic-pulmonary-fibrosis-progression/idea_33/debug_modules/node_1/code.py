import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import LungDataset, get_transforms
from library.model import GTVRNet
from library.train import LaplaceLikelihoodLoss, train_epoch, valid_epoch


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Overrides for Speed and Stability
    print("\n[Step 1] Configuring environment...")
    # Disable downloading pretrained weights to ensure offline execution
    Config.PRETRAINED = False
    # Use a tiny subset of data for demonstration
    Config.DEBUG_SAMPLE_SIZE = 10
    # Reduce training parameters
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure directories exist
    Config.make_dirs()

    # Set seeds
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")
    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # 2. Data Pipeline Verification
    print("\n[Step 2] Verifying Data Pipeline...")

    # Initialize Dataset
    train_dataset = LungDataset(
        csv_path=Config.TRAIN_CSV,
        mode="train",
        transform=get_transforms("train"),
        load_cached_data=True,
    )

    # Initialize Loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    img_axial = batch["img_axial"].to(device)
    img_coronal = batch["img_coronal"].to(device)
    tabular = batch["tabular"].to(device)
    meta = batch["meta"].to(device)
    target = batch["target"].to(device)

    print("Batch keys:", batch.keys())

    # Assertions for Data Shapes
    # Images: (B, 3, 224, 224)
    assert img_axial.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Incorrect Axial shape: {img_axial.shape}"
    assert img_coronal.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Incorrect Coronal shape: {img_coronal.shape}"
    # Tabular: (B, 6) -> Age, Sex, Smoke_Ex, Smoke_Never, Smoke_Current, BasePercent
    assert tabular.shape == (
        Config.BATCH_SIZE,
        6,
    ), f"Incorrect Tabular shape: {tabular.shape}"
    # Meta: (B, 2) -> Baseline_FVC, Delta_Week
    assert meta.shape == (Config.BATCH_SIZE, 2), f"Incorrect Meta shape: {meta.shape}"
    # Target: (B, 1)
    assert target.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect Target shape: {target.shape}"

    print("Data Pipeline Verification Passed.")

    # 3. Model Architecture Verification
    print("\n[Step 3] Verifying Model Architecture...")

    model = GTVRNet().to(device)

    # Forward Pass
    fvc_pred, sigma_pred = model(img_axial, img_coronal, tabular, meta)

    # Reshape predictions to match target (B, 1)
    fvc_pred = fvc_pred.view(-1, 1)
    sigma_pred = sigma_pred.view(-1, 1)

    # Assertions for Output
    assert fvc_pred.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect FVC Pred shape: {fvc_pred.shape}"
    assert sigma_pred.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect Sigma Pred shape: {sigma_pred.shape}"

    # Check positivity of sigma (confidence)
    if not torch.all(sigma_pred > 0):
        raise AssertionError("Model produced non-positive confidence values (sigma).")

    print(f"Model Output Shapes: FVC {fvc_pred.shape}, Sigma {sigma_pred.shape}")
    print("Model Architecture Verification Passed.")

    # 4. Loss and Metric Verification
    print("\n[Step 4] Verifying Loss and Metric...")

    criterion = LaplaceLikelihoodLoss()

    # Calculate Loss
    # Reshape predictions to match target for loss calculation if needed,
    # though the model output is already (B, 1) and target is (B, 1).
    loss = criterion(fvc_pred, sigma_pred, target)

    # Calculate Metric
    metric_score = laplace_log_likelihood_metric(target, fvc_pred, sigma_pred)

    print(f"Calculated Loss: {loss.item():.4f}")
    print(f"Calculated Metric: {metric_score:.4f}")

    assert torch.isfinite(loss), "Loss is not finite."
    assert np.isfinite(metric_score), "Metric is not finite."

    print("Loss and Metric Verification Passed.")

    # 5. Training Loop Integration
    print("\n[Step 5] Running Training Loop Demonstration...")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Create a validation loader
    val_dataset = LungDataset(
        csv_path=Config.VAL_CSV,
        mode="val",
        transform=get_transforms("val"),
        load_cached_data=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run loop
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metric = valid_epoch(model, val_loader, device)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Metric: {val_metric:.4f}"
        )

    print("Training Loop Demonstration Passed.")

    # 6. Inference Demonstration
    print("\n[Step 6] Running Inference Demonstration...")

    test_dataset = LungDataset(
        csv_path=Config.TEST_CSV,
        mode="test",
        transform=get_transforms("val"),
        load_cached_data=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    model.eval()
    results = []

    with torch.no_grad():
        # Just run one batch for demo
        batch = next(iter(test_loader))

        img_axial = batch["img_axial"].to(device)
        img_coronal = batch["img_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        meta = batch["meta"].to(device)
        patient_weeks = batch["patient_week"]

        fvc_pred, sigma_pred = model(img_axial, img_coronal, tabular, meta)

        # Format results
        for i in range(len(patient_weeks)):
            results.append(
                {
                    "Patient_Week": patient_weeks[i],
                    "FVC": fvc_pred[i].item(),
                    "Confidence": sigma_pred[i].item(),
                }
            )

    print("Sample Inference Results:")
    print(pd.DataFrame(results))

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
