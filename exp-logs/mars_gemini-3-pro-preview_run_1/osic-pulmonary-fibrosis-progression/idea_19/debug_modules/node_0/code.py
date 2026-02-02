import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.dataset import LungDataset
from library.model import TabularGatedDualViewNetwork
from library.engine import fit, criterion


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Override for Speed
    # We modify the Config class attributes directly to run a fast demo
    print("Configuring parameters for fast demonstration...")
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.IDEA_NAME = "demo_run"

    # Update paths to ensure we don't overwrite existing best models during demo
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "demo_run", "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "demo_run")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Initialize directories
    Config.setup()

    # Set Seed
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Preparation
    print("\n=== Initializing Datasets ===")

    # Initialize Datasets
    # Note: The DicomProcessor will attempt to cache processed images.
    # Since we are running a demo, we will subset the dataframes immediately
    # after initialization to avoid processing thousands of DICOMs.

    train_dataset = LungDataset(mode="train", load_cached_data=True)
    val_dataset = LungDataset(mode="val", load_cached_data=True)
    test_dataset = LungDataset(mode="test", load_cached_data=True)

    # Subset Datasets for Speed (e.g., top 32 samples for train, 16 for val, 16 for test)
    # This ensures the code runs in minutes, not hours.
    subset_size_train = 32
    subset_size_val = 16
    subset_size_test = 16

    print(
        f"Subsetting datasets: Train={subset_size_train}, Val={subset_size_val}, Test={subset_size_test}"
    )
    train_dataset.df = train_dataset.df.iloc[:subset_size_train].copy()
    val_dataset.df = val_dataset.df.iloc[:subset_size_val].copy()
    test_dataset.df = test_dataset.df.iloc[:subset_size_test].copy()

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Validation: Check a single batch
    print("\n=== Validating Data Loading ===")
    sample_batch = next(iter(train_loader))

    # Assert shapes
    assert sample_batch["image_axial"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect Axial Image Shape"
    assert sample_batch["image_coronal"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect Coronal Image Shape"
    assert sample_batch["tabular"].shape == (
        Config.BATCH_SIZE,
        7,
    ), "Incorrect Tabular Shape"
    assert sample_batch["dt"].shape == (Config.BATCH_SIZE,), "Incorrect dt Shape"

    print("Data shapes verified successfully.")

    # 3. Model Initialization
    print("\n=== Initializing Model ===")
    model = TabularGatedDualViewNetwork()
    model.to(device)

    # Validation: Forward pass with sample batch
    print("Running dummy forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            sample_batch["image_axial"].to(device),
            sample_batch["image_coronal"].to(device),
            sample_batch["tabular"].to(device),
            sample_batch["dt"].to(device),
            sample_batch["baseline_fvc"].to(device),
        )

    # Assert Output Keys and Shapes
    assert "fvc_pred" in outputs
    assert "confidence_pred" in outputs
    assert outputs["fvc_pred"].shape == (Config.BATCH_SIZE,)
    assert outputs["confidence_pred"].shape == (Config.BATCH_SIZE,)
    assert not torch.isnan(
        outputs["fvc_pred"]
    ).any(), "Model produced NaNs in FVC prediction"

    # Validate Loss Calculation
    loss = criterion(
        sample_batch["target"].to(device),
        outputs["fvc_pred"],
        outputs["confidence_pred"],
        device,
    )
    assert torch.isfinite(loss), "Loss is not finite"
    print(f"Dummy forward pass successful. Initial Loss: {loss.item():.4f}")

    # 4. Training Loop
    print("\n=== Starting Training Loop ===")
    # fit() handles the loop, validation, and checkpointing
    best_score = fit(
        model,
        train_loader,
        val_loader,
        device,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
    )

    print(f"Training finished. Best Validation Score: {best_score:.4f}")

    # Check if model file was created
    if not os.path.exists(Config.BEST_MODEL_PATH):
        # If training was too short or score didn't improve (unlikely with random init vs valid),
        # we might not have a file. For demo purposes, save current model.
        torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        print("Saved current model as best model (fallback).")
    else:
        print(f"Verified checkpoint exists at {Config.BEST_MODEL_PATH}")

    # 5. Inference & Submission
    print("\n=== Generating Submission ===")

    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    submission_rows = []

    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tab = batch["tabular"].to(device)
            dt = batch["dt"].to(device)
            base_fvc = batch["baseline_fvc"].to(device)
            patient_ids = batch["patient_id"]  # List of strings

            # Predict
            out = model(img_ax, img_cor, tab, dt, base_fvc)

            fvc_preds = out["fvc_pred"].cpu().numpy()
            conf_preds = out["confidence_pred"].cpu().numpy()

            # Reconstruct Patient_Week ID
            # We need the week number. In test dataset, dt = Predict_Week - Baseline_Week.
            # We can retrieve Predict_Week from the dataframe, but the DataLoader shuffles/batches.
            # However, test_loader shuffle is False.
            # Let's rely on the fact that we can reconstruct the ID if we had the week.
            # The dataset doesn't yield 'Predict_Week' directly in __getitem__.
            # Strategy: The test.csv in metadata has 'Patient_Week' column.
            # Since we didn't shuffle test_loader, we can match indices if we iterate carefully,
            # OR we can modify dataset to return Patient_Week.
            # Given we cannot modify library files, we will map via the dataframe index.
            # But DataLoader batching complicates direct index mapping.

            # Alternative: The submission format requires Patient_Week.
            # The provided Dataset class returns 'patient_id' and 'dt'.
            # We can calculate Week = Baseline_Week + dt.
            # But we don't have Baseline_Week in the batch return.

            # Workaround for Demo:
            # We will iterate the dataframe slice directly to get metadata,
            # and assume the loader order is preserved (shuffle=False).
            pass

    # Re-implementing inference loop to ensure alignment with metadata
    # We iterate the dataset by index to ensure we get the correct Patient_Week string
    # from the dataframe, while using the model for prediction.

    test_df_slice = test_dataset.df  # This is the sliced dataframe
    predictions = []
    confidences = []

    # Process in batches manually to ensure alignment or use loader and zip with df
    # Since we set shuffle=False, the loader yields data in the order of the dataframe.

    current_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            batch_size = batch["image_axial"].shape[0]

            # Move to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tab = batch["tabular"].to(device)
            dt = batch["dt"].to(device)
            base_fvc = batch["baseline_fvc"].to(device)

            out = model(img_ax, img_cor, tab, dt, base_fvc)

            predictions.extend(out["fvc_pred"].cpu().numpy())
            confidences.extend(out["confidence_pred"].cpu().numpy())

            current_idx += batch_size

    # Assign to dataframe
    test_df_slice["FVC"] = predictions
    test_df_slice["Confidence"] = confidences

    # Prepare final submission dataframe
    submission_df = test_df_slice[["Patient_Week", "FVC", "Confidence"]].copy()

    # Save
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print("First 5 rows:")
    print(submission_df.head())

    # Final Verification
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found"
    assert len(submission_df) == subset_size_test, "Submission row count mismatch"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
