import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, score_function
from library.data import get_dataloaders
from library.model import UDSRNet
from library.train import (
    train_one_epoch,
    validate,
    generate_submission,
    LaplaceLogLikelihoodLoss,
)


def create_subset_metadata(base_dir, subset_dir, n_patients=2):
    """
    Creates a tiny subset of the metadata to speed up the demonstration.
    """
    os.makedirs(subset_dir, exist_ok=True)

    # 1. Train Subset
    train_df = pd.read_csv(os.path.join(base_dir, "train.csv"))
    train_patients = train_df["Patient"].unique()[:n_patients]
    train_subset = train_df[train_df["Patient"].isin(train_patients)].copy()
    train_path = os.path.join(subset_dir, "train.csv")
    train_subset.to_csv(train_path, index=False)

    # 2. Val Subset
    val_df = pd.read_csv(os.path.join(base_dir, "val.csv"))
    val_patients = val_df["Patient"].unique()[:n_patients]
    val_subset = val_df[val_df["Patient"].isin(val_patients)].copy()
    val_path = os.path.join(subset_dir, "val.csv")
    val_subset.to_csv(val_path, index=False)

    # 3. Test Subset
    test_df = pd.read_csv(os.path.join(base_dir, "test.csv"))
    test_patients = test_df["Patient"].unique()[:n_patients]
    test_subset = test_df[test_df["Patient"].isin(test_patients)].copy()
    test_path = os.path.join(subset_dir, "test.csv")
    test_subset.to_csv(test_path, index=False)

    # 4. Sample Submission Subset
    # We need to filter the sample submission to only include rows for our test subset patients
    sub_df = pd.read_csv(
        Config.SAMPLE_SUBMISSION
    )  # Original sample submission path from Config
    # Extract patient ID from Patient_Week string (Format: ID..._Week)
    sub_df["Patient_ID"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub_subset = (
        sub_df[sub_df["Patient_ID"].isin(test_patients)]
        .drop(columns=["Patient_ID"])
        .copy()
    )
    sub_path = os.path.join(subset_dir, "sample_submission.csv")
    sub_subset.to_csv(sub_path, index=False)

    return train_path, val_path, test_path, sub_path


if __name__ == "__main__":
    # =========================================================================
    # 1. Setup & Configuration Overrides
    # =========================================================================
    print("--- Setting up environment and configuration ---")

    # Set seeds for reproducibility
    seed_everything(42)

    # Override Config for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.IMG_SIZE = 64  # Small image size for fast processing
    Config.SLICES_PER_PATIENT = 3

    # Define working directories for this demo
    DEMO_DIR = "./working/demo_execution"
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Setup directories
    Config.setup()

    # Create subset metadata to run pipeline quickly
    METADATA_SUBSET_DIR = os.path.join(DEMO_DIR, "metadata_subset")
    train_csv, val_csv, test_csv, sub_csv = create_subset_metadata(
        Config.METADATA_DIR, METADATA_SUBSET_DIR, n_patients=3
    )

    # Override Config paths to point to subsets
    Config.TRAIN_CSV = train_csv
    Config.VAL_CSV = val_csv
    Config.TEST_CSV = test_csv
    Config.SAMPLE_SUBMISSION = sub_csv
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    print(f"Configuration updated. Using subset metadata at {METADATA_SUBSET_DIR}")

    # =========================================================================
    # 2. Data Loading & Caching
    # =========================================================================
    print("\n--- Initializing Data Loaders ---")

    # This will trigger image caching (resizing/windowing) and scaler fitting
    train_loader, val_loader, test_loader, scalers = get_dataloaders(
        load_cached_data=True
    )

    # Verify cache generation
    cached_files = os.listdir(Config.CACHE_DIR)
    print(f"Cached files count: {len(cached_files)}")
    assert len(cached_files) > 0, "Image caching failed, no files found."

    # Verify DataLoader output structure
    batch = next(iter(train_loader))
    images = batch["image"]
    tabular = batch["tabular"]
    targets = batch["fvc_target"]

    print(
        f"Batch Shapes -> Image: {images.shape}, Tabular: {tabular.shape}, Target: {targets.shape}"
    )

    # Assertions for shapes
    # Image: (B, Slices, H, W) -> (2, 3, 64, 64)
    assert images.shape == (
        Config.BATCH_SIZE,
        Config.SLICES_PER_PATIENT,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    # Tabular: (B, 5) -> [Baseline_FVC, Rel_Time, Age, Sex, Smoking]
    assert tabular.shape == (Config.BATCH_SIZE, 5)

    # =========================================================================
    # 3. Model Instantiation & Forward Pass
    # =========================================================================
    print("\n--- Initializing Model ---")

    device = torch.device("cpu")  # Use CPU for simple verification
    model = UDSRNet().to(device)

    # Perform a forward pass
    print("Running forward pass...")
    mu, sigma = model(images.to(device), tabular.to(device))

    print(f"Output Shapes -> Mu: {mu.shape}, Sigma: {sigma.shape}")

    # Assertions for model output
    assert mu.shape == (Config.BATCH_SIZE,)
    assert sigma.shape == (Config.BATCH_SIZE,)
    assert torch.all(sigma > 0), "Sigma must be positive (Softplus)"

    # =========================================================================
    # 4. Loss Function Verification
    # =========================================================================
    print("\n--- Verifying Loss Function ---")

    criterion = LaplaceLogLikelihoodLoss()
    loss = criterion(mu, sigma, targets.to(device))

    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    # =========================================================================
    # 5. Training Loop Simulation
    # =========================================================================
    print("\n--- Simulating Training Loop ---")

    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss: {train_loss:.4f}")

    # Validate
    target_scaler = scalers["target_scaler"]
    val_loss, val_score = validate(model, val_loader, criterion, device, target_scaler)
    print(f"Val Loss: {val_loss:.4f} | Val Score: {val_score:.4f}")

    # Save "best" model manually for the inference step
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    assert os.path.exists(Config.BEST_MODEL_PATH), "Model checkpoint not saved."

    # =========================================================================
    # 6. Inference & Submission Generation
    # =========================================================================
    print("\n--- Generating Submission ---")

    # Generate submission using the saved model
    # We reload the model to ensure the loading mechanism works as expected
    model_inference = UDSRNet().to(device)
    model_inference.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=device)
    )

    generate_submission(model_inference, scalers, device)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(submission_df.head())

    # Check columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(submission_df.columns)}"

    # Check values
    assert not submission_df["FVC"].isnull().any(), "NaN found in FVC predictions"
    assert (
        not submission_df["Confidence"].isnull().any()
    ), "NaN found in Confidence predictions"
    assert (
        submission_df["Confidence"] >= 70
    ).all(), "Confidence clipping failed (values < 70 found)"

    print("\n--- Demo Completed Successfully ---")
