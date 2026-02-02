import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# Ensure the current directory is in the path for imports
sys.path.append(os.getcwd())

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, count_parameters, get_logger
from library.dataset import LungDataset
from library.model import NSLHN
from library.loss import LaplaceLogLikelihoodLoss
from library.engine import fit


def main():
    print("Starting Demo Script...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Suppress warnings for clean output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    seed_everything(42)

    # Modify Config for fast demonstration
    print("\n[1] Configuring environment for rapid demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SIZE = 16  # Use a tiny subset of data
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.print_config()

    # Initialize Logger
    logger = get_logger()

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("[2] Verifying Data Pipeline...")

    # Instantiate Datasets
    train_dataset = LungDataset(mode="train", split="train")
    val_dataset = LungDataset(mode="inference", split="val")

    print(f"    Train Dataset Size (Debug): {len(train_dataset)}")
    print(f"    Val Dataset Size (Debug): {len(val_dataset)}")

    # Verify a single sample
    sample = train_dataset[0]

    # Check keys
    required_keys = [
        "axial",
        "coronal",
        "tabular",
        "fvc_target",
        "base_fvc",
        "delta_week",
        "patient_week_id",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    # Check Shapes
    # Images: (3, 224, 224)
    assert sample["axial"].shape == (
        3,
        224,
        224,
    ), f"Incorrect axial shape: {sample['axial'].shape}"
    assert sample["coronal"].shape == (
        3,
        224,
        224,
    ), f"Incorrect coronal shape: {sample['coronal'].shape}"
    # Tabular: (6,)
    assert sample["tabular"].shape == (
        6,
    ), f"Incorrect tabular shape: {sample['tabular'].shape}"
    # Scalars
    assert isinstance(sample["fvc_target"], torch.Tensor), "Target is not a tensor"

    print("    Data loading and shape verification passed.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    device = Config.DEVICE
    model = NSLHN().to(device)

    # Count parameters
    num_params = count_parameters(model)
    print(f"    Model instantiated. Trainable Parameters: {num_params:,}")

    # Create a dummy batch based on the sample we fetched
    batch_size = 2
    dummy_axial = torch.stack([sample["axial"]] * batch_size).to(device)
    dummy_coronal = torch.stack([sample["coronal"]] * batch_size).to(device)
    dummy_tabular = torch.stack([sample["tabular"]] * batch_size).to(device)
    dummy_base_fvc = torch.stack([sample["base_fvc"]] * batch_size).to(device)
    dummy_delta_week = torch.stack([sample["delta_week"]] * batch_size).to(device)

    # Forward Pass
    model.eval()
    with torch.no_grad():
        pred_fvc, pred_sigma = model(
            dummy_axial, dummy_coronal, dummy_tabular, dummy_base_fvc, dummy_delta_week
        )

    # Verify Output Shapes
    assert pred_fvc.shape == (
        batch_size,
    ), f"Expected FVC shape {(batch_size,)}, got {pred_fvc.shape}"
    assert pred_sigma.shape == (
        batch_size,
    ), f"Expected Sigma shape {(batch_size,)}, got {pred_sigma.shape}"

    # Verify Values (No NaNs)
    assert not torch.isnan(pred_fvc).any(), "Model produced NaNs in FVC prediction"
    assert not torch.isnan(pred_sigma).any(), "Model produced NaNs in Sigma prediction"

    print("    Forward pass successful. Output shapes correct.")

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print("\n[4] Verifying Loss Function...")

    criterion = LaplaceLogLikelihoodLoss()
    dummy_target = torch.tensor([2500.0, 2600.0], device=device)

    # Calculate loss
    loss = criterion(pred_fvc, pred_sigma, dummy_target)

    # Check validity
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"

    print(f"    Loss calculation successful. Value: {loss.item():.4f}")

    # ==========================================
    # 5. Training Loop Integration (Engine)
    # ==========================================
    print("\n[5] Verifying Training Engine...")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,  # Drop last to ensure batch norm works if batch size is small
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Run Training
    print("    Starting training loop (2 epochs)...")
    best_score = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
    )

    assert isinstance(best_score, float), "Fit function did not return a float score"
    print(f"    Training complete. Best Validation Score: {best_score:.4f}")

    # ==========================================
    # 6. Inference & Submission Check
    # ==========================================
    print("\n[6] Verifying Inference on Test Set...")

    # Load Test Dataset
    test_dataset = LungDataset(mode="inference", split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    print(f"    Test Dataset Size (Debug): {len(test_dataset)}")

    model.eval()
    predictions = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Move inputs
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            delta_week = batch["delta_week"].to(device)
            patient_week_ids = batch["patient_week_id"]

            # Predict
            pred_fvc, pred_sigma = model(axial, coronal, tabular, base_fvc, delta_week)

            # Collect results
            for j in range(len(patient_week_ids)):
                predictions.append(
                    {
                        "Patient_Week": patient_week_ids[j],
                        "FVC": pred_fvc[j].item(),
                        "Confidence": pred_sigma[j].item(),
                    }
                )

            # Stop after one batch for demo speed
            break

    # Verify prediction format
    assert len(predictions) > 0, "No predictions generated"
    first_pred = predictions[0]
    assert "Patient_Week" in first_pred
    assert "FVC" in first_pred
    assert "Confidence" in first_pred

    print("    Inference successful. Sample Prediction:")
    print(f"    {first_pred}")

    print("\n" + "=" * 40)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    main()
