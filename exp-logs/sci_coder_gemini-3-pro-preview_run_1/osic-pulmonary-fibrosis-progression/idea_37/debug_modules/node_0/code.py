import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, score_function
from library.data import get_dataloaders
from library.model import DALANet
from library.train import train_one_epoch, valid_one_epoch, LaplaceLogLikelihoodLoss

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_subset_data(n_samples=32):
    """
    Creates small subsets of the metadata CSVs for rapid demonstration.
    """
    print(f"Creating data subsets (N={n_samples})...")

    # Load original metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Create subsets
    train_subset = train_df.head(n_samples).copy()
    val_subset = val_df.head(n_samples).copy()
    test_subset = test_df.head(n_samples).copy()

    # Define paths for subsets
    subset_train_path = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    subset_val_path = os.path.join(Config.WORKING_DIR, "val_subset.csv")
    subset_test_path = os.path.join(Config.WORKING_DIR, "test_subset.csv")

    # Save subsets
    train_subset.to_csv(subset_train_path, index=False)
    val_subset.to_csv(subset_val_path, index=False)
    test_subset.to_csv(subset_test_path, index=False)

    return subset_train_path, subset_val_path, subset_test_path


def main():
    # 1. Setup
    print("Initializing Demonstration...")
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Create working directory if it doesn't exist (Config.setup() does this, but we do it explicitly for the subsets)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Configure for Speed (Override Config defaults)
    train_csv, val_csv, test_csv = create_subset_data(n_samples=16)

    # Monkey-patch the Config class to use subsets and lightweight settings
    Config.TRAIN_CSV = train_csv
    Config.VAL_CSV = val_csv
    Config.TEST_CSV = test_csv
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(
        f"Configuration updated: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}"
    )

    # 3. Verify Data Loading
    print("\n--- Verifying Data Loading ---")
    train_loader, val_loader, test_loader = get_dataloaders(Config)

    # Fetch one batch
    batch = next(iter(train_loader))

    axial = batch["axial"].to(device)
    coronal = batch["coronal"].to(device)
    tabular = batch["tabular"].to(device)
    target = batch["target"].to(device)
    meta = batch["meta"]

    # Assert Shapes
    # Expected: (B, 3, 240, 240) for images, (B, 7) for tabular
    print(f"Axial Shape: {axial.shape}")
    print(f"Coronal Shape: {coronal.shape}")
    print(f"Tabular Shape: {tabular.shape}")

    assert axial.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Axial image shape mismatch"
    assert coronal.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Coronal image shape mismatch"
    assert tabular.shape == (Config.BATCH_SIZE, 7), "Tabular feature shape mismatch"
    assert target.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch"

    print("Data Loading Verified.")

    # 4. Verify Model Architecture & Forward Pass
    print("\n--- Verifying Model Architecture ---")
    model = DALANet().to(device)

    # Prepare scalar inputs from meta
    delta_week = (
        torch.tensor(meta["Delta_Week"], dtype=torch.float32).to(device).view(-1, 1)
    )
    baseline_fvc = (
        torch.tensor(meta["Baseline_FVC"], dtype=torch.float32).to(device).view(-1, 1)
    )

    # Forward Pass
    fvc_pred, sigma_pred = model(axial, coronal, tabular, delta_week, baseline_fvc)

    print(f"Prediction Shape: {fvc_pred.shape}")
    print(f"Confidence Shape: {sigma_pred.shape}")

    # Assertions
    assert fvc_pred.shape == (Config.BATCH_SIZE, 1), "FVC prediction shape mismatch"
    assert sigma_pred.shape == (Config.BATCH_SIZE, 1), "Sigma prediction shape mismatch"

    # Verify constraints (Sigma must be positive due to Softplus)
    if (sigma_pred <= 0).any():
        raise AssertionError(
            "Model produced non-positive confidence values (Sigma <= 0)."
        )

    print("Model Forward Pass Verified.")

    # 5. Verify Loss Calculation
    print("\n--- Verifying Loss Function ---")
    criterion = LaplaceLogLikelihoodLoss()
    target_view = target.view(-1, 1)

    loss = criterion(fvc_pred, target_view, sigma_pred)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    print("Loss Function Verified.")

    # 6. Training Loop Demonstration
    print("\n--- Running Training Loop (Demo) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_score = valid_one_epoch(model, val_loader, device)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.4f}"
        )

    print("Training Loop Verified.")

    # 7. Inference & Submission Generation
    print("\n--- Running Inference & Generating Submission ---")
    model.eval()
    submission_rows = []

    with torch.no_grad():
        for batch in test_loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)

            # Meta data for test
            delta_week = (
                torch.tensor(batch["meta"]["Delta_Week"], dtype=torch.float32)
                .to(device)
                .view(-1, 1)
            )
            baseline_fvc = (
                torch.tensor(batch["meta"]["Baseline_FVC"], dtype=torch.float32)
                .to(device)
                .view(-1, 1)
            )
            patient_weeks = batch["meta"]["Patient_Week"]

            # Predict
            fvc_pred, sigma_pred = model(
                axial, coronal, tabular, delta_week, baseline_fvc
            )

            # Collect results
            fvc_np = fvc_pred.cpu().numpy().flatten()
            sigma_np = sigma_pred.cpu().numpy().flatten()

            for pw, fvc, sigma in zip(patient_weeks, fvc_np, sigma_np):
                submission_rows.append(
                    {"Patient_Week": pw, "FVC": fvc, "Confidence": sigma}
                )

    # Create DataFrame
    sub_df = pd.DataFrame(submission_rows)

    # Verify Submission Format
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in sub_df.columns for col in expected_cols
    ), "Submission missing required columns"
    assert len(sub_df) == len(
        pd.read_csv(Config.TEST_CSV)
    ), "Submission row count mismatch"

    # Save dummy submission
    demo_sub_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    sub_df.to_csv(demo_sub_path, index=False)

    print(f"Submission generated with {len(sub_df)} rows.")
    print(sub_df.head())
    print("\nDemonstration Complete. All checks passed.")


if __name__ == "__main__":
    main()
