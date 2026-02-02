import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config, seed_everything
from library.data import FibrosisDataset, get_dataloaders
from library.model import SiameseDualAxisNet
from library.train import CombinedLoss, train_one_epoch, validate_one_epoch
from library.utils import score_function


def main():
    print("Initializing Demonstration Script...")

    # 1. Configuration Overrides for Speed and Demonstration
    # We modify the Config class attributes directly to affect the imported modules
    print("Configuring environment for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SIZE = 12  # Small subset for quick processing
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Use main process for simplicity
    Config.PRETRAINED = False  # Skip downloading weights for speed
    Config.CACHE_DIR = "./working/demo_cache"  # Ensure we write to working dir
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)

    # 2. Data Loading Demonstration
    print("\n--- 1. Data Loading Verification ---")

    # Load metadata
    try:
        train_df = pd.read_csv(Config.TRAIN_CSV)
        val_df = pd.read_csv(Config.VAL_CSV)
        test_df = pd.read_csv(Config.TEST_CSV)
    except FileNotFoundError:
        print("Metadata files not found. Ensure metadata generation script ran.")
        return

    # Initialize Dataset
    # We use 'train' mode to check target generation
    # load_cache=False forces processing (or zero-generation if pydicom missing)
    ds = FibrosisDataset(
        train_df.head(Config.DEBUG_SIZE), mode="train", transform=None, load_cache=False
    )

    # Fetch one sample
    sample_inputs, sample_targets = ds[0]
    axial, coronal, tabular = sample_inputs

    # Verification
    print(f"Sample Axial Shape: {axial.shape}")
    print(f"Sample Coronal Shape: {coronal.shape}")
    print(f"Sample Tabular Shape: {tabular.shape}")
    print(f"Sample Targets Keys: {list(sample_targets.keys())}")

    # Assertions
    assert axial.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect Axial Image Shape"
    assert coronal.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect Coronal Image Shape"
    assert tabular.shape == (4,), "Incorrect Tabular Feature Shape"
    assert "fvc" in sample_targets, "Target 'fvc' missing"

    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, batch_size=Config.BATCH_SIZE
    )

    # Fetch one batch
    batch_inputs, batch_targets = next(iter(train_loader))
    b_axial, b_coronal, b_tabular = batch_inputs

    print(f"Batch Axial Shape: {b_axial.shape}")
    assert b_axial.shape[0] == Config.BATCH_SIZE, "Incorrect Batch Size"

    # 3. Model Initialization and Forward Pass
    print("\n--- 2. Model Architecture Verification ---")

    device = Config.DEVICE
    print(f"Running on device: {device}")
    model = SiameseDualAxisNet().to(device)

    # Move batch to device
    b_axial = b_axial.to(device)
    b_coronal = b_coronal.to(device)
    b_tabular = b_tabular.to(device)

    # Forward pass
    outputs = model(b_axial, b_coronal, b_tabular)

    print("Model Output Keys:", list(outputs.keys()))

    # Verify Outputs
    assert "alpha" in outputs
    assert "sigma_base" in outputs
    assert "sigma_growth" in outputs
    assert "pred_percent" in outputs
    assert outputs["alpha"].shape[0] == Config.BATCH_SIZE

    # 4. Loss Calculation
    print("\n--- 3. Loss Function Verification ---")

    criterion = CombinedLoss().to(device)

    # Move targets to device
    b_targets = {k: v.to(device) for k, v in batch_targets.items()}

    loss, metric_loss, aux_loss = criterion(outputs, b_targets)

    print(f"Total Loss: {loss.item():.4f}")
    print(f"Metric Loss component: {metric_loss.item():.4f}")
    print(f"Aux Loss component: {aux_loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"

    # 5. Training Loop Simulation
    print("\n--- 4. Training Loop Simulation ---")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Train one epoch (using the small debug loader)
    t_loss, t_m_loss, t_a_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )
    print(f"Train Epoch Result -> Loss: {t_loss:.4f}")

    # Validate one epoch
    v_score = validate_one_epoch(model, val_loader, device)
    print(f"Validation Score: {v_score:.4f}")

    # 6. Metric Logic Verification
    print("\n--- 5. Metric Logic Verification ---")
    # Test case: Perfect prediction
    # Delta = 0, Sigma = 70 (clipped min)
    # Metric = - (sqrt(2)*0)/70 - ln(sqrt(2)*70) = - ln(98.99) approx -4.595
    y_true = np.array([2000])
    y_pred = np.array([2000])
    sigma = np.array([10])  # Should be clipped to 70

    score = score_function(y_true, y_pred, sigma)
    expected_score = -np.log(np.sqrt(2) * 70)
    print(f"Calculated Score (Perfect Pred): {score:.4f}")
    print(f"Expected Score: {expected_score:.4f}")

    assert np.isclose(score, expected_score, atol=1e-4), "Metric calculation mismatch"

    # 7. Inference and Submission
    print("\n--- 6. Inference and Submission Generation ---")

    model.eval()
    submission_rows = []

    with torch.no_grad():
        for inputs, meta in test_loader:
            axial_img, coronal_img, tabular = inputs
            axial_img = axial_img.to(device)
            coronal_img = coronal_img.to(device)
            tabular = tabular.to(device)

            # Metadata for reconstruction
            base_fvc = meta["base_fvc"].to(device)
            week_delta = meta["week_delta"].to(device)
            patient_weeks = meta["patient_week"]

            # Forward
            outputs = model(axial_img, coronal_img, tabular)

            alpha = outputs["alpha"]
            sigma_base = outputs["sigma_base"]
            sigma_growth = outputs["sigma_growth"]

            # Predict
            fvc_pred = base_fvc + alpha * week_delta
            confidence = sigma_base + sigma_growth * torch.abs(week_delta)

            # Collect
            fvc_pred = fvc_pred.cpu().numpy()
            confidence = confidence.cpu().numpy()

            for pw, fvc, conf in zip(patient_weeks, fvc_pred, confidence):
                submission_rows.append(
                    {"Patient_Week": pw, "FVC": fvc, "Confidence": conf}
                )

    # Create DataFrame
    sub_df = pd.DataFrame(submission_rows)
    print(f"Generated {len(sub_df)} predictions.")
    print("Sample predictions:")
    print(sub_df.head())

    # Save (Mock submission)
    sub_df.to_csv("submission_demo.csv", index=False)
    print("Saved submission_demo.csv")

    print("\nDemonstration Complete.")


if __name__ == "__main__":
    main()
