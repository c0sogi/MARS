import os
import torch
import numpy as np
import pandas as pd
import sys

# Import provided library modules
from library.config import Config, seed_everything
from library.data import get_dataloaders, TabularPreprocessor
from library.model import SPPDSNet
from library.train import train_model, MetricAlignedLaplaceLoss
from library.utils import calculate_metric, save_submission


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")
    # Override Config values to run a fast demo
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.DEBUG_SIZE = 12  # Use only 12 samples for train/val
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seed
    seed_everything(Config.SEED)
    print("Configuration updated: EPOCHS=1, BATCH_SIZE=4, DEBUG_SIZE=12")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Preprocessing Demo
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Loading and Preprocessing...")

    # Get dataloaders in debug mode
    train_loader, val_loader, test_loader, preprocessor = get_dataloaders(debug=True)

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches:   {len(val_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    images = batch["image"]
    tabular = batch["tabular"]
    target = batch["target"]
    target_scaled = batch["target_scaled"]
    patient_ids = batch["patient_week"]

    print(f"Batch Keys: {list(batch.keys())}")
    print(f"Image Shape: {images.shape} (Expected: [{Config.BATCH_SIZE}, 3, 260, 260])")
    print(f"Tabular Shape: {tabular.shape} (Expected: [{Config.BATCH_SIZE}, 6])")
    print(f"Target Shape: {target.shape}")

    # Assertions to verify data integrity
    assert images.ndim == 4, "Images should be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels (slices)"
    assert tabular.shape[1] == 6, "Tabular data should have 6 features"
    assert not torch.isnan(images).any(), "Images contain NaNs"
    assert not torch.isnan(tabular).any(), "Tabular data contains NaNs"

    print("Data loading verification successful.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass Demo
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Model Initialization and Forward Pass...")

    device = Config.DEVICE
    model = SPPDSNet().to(device)

    # Move batch to device
    images_dev = images.to(device)
    tabular_dev = tabular.to(device)

    # Forward pass
    mu, sigma = model(images_dev, tabular_dev)

    print(f"Output Mu Shape: {mu.shape}")
    print(f"Output Sigma Shape: {sigma.shape}")

    # Assertions
    assert mu.shape == (Config.BATCH_SIZE, 1), "Mu output shape mismatch"
    assert sigma.shape == (Config.BATCH_SIZE, 1), "Sigma output shape mismatch"
    assert (sigma > 0).all(), "Sigma (uncertainty) must be positive"

    print("Forward pass verification successful.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Demo
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Loss Calculation...")

    criterion = MetricAlignedLaplaceLoss()

    # Squeeze outputs to match target shape if necessary
    loss = criterion(mu.squeeze(), sigma.squeeze(), target_scaled.to(device).squeeze())

    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Infinite"

    print("Loss calculation verification successful.")

    # -------------------------------------------------------------------------
    # 5. Full Training Loop Demo
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (Debug Mode)...")

    # This uses the library.train.train_model function
    # It will train for Config.EPOCHS (set to 1 above) on the debug subset
    best_score = train_model(debug=True)

    print(f"Training demo complete. Best Validation Score: {best_score}")
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."

    # -------------------------------------------------------------------------
    # 6. Inference & Submission Demo
    # -------------------------------------------------------------------------
    print("\n[6] Demonstrating Inference and Submission Generation...")

    # Load the best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    predictions = []

    # Get scaler std for inverse transforming sigma
    # FVC is the first feature fitted in target_scaler
    fvc_scale = preprocessor.target_scaler.scale_[0]

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            tabs = batch["tabular"].to(device)
            p_weeks = batch["patient_week"]

            # Predict
            pred_mu_scaled, pred_sigma_scaled = model(imgs, tabs)

            # Move to CPU numpy
            pred_mu_scaled = pred_mu_scaled.cpu().numpy().flatten()
            pred_sigma_scaled = pred_sigma_scaled.cpu().numpy().flatten()

            # Inverse Transform
            # 1. FVC: Standard inverse transform
            pred_fvc = preprocessor.inverse_transform_target(pred_mu_scaled).flatten()

            # 2. Confidence: Scale by std deviation
            pred_conf = pred_sigma_scaled * fvc_scale

            # Store
            for pw, fvc, conf in zip(p_weeks, pred_fvc, pred_conf):
                predictions.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    # Create DataFrame
    sub_df = pd.DataFrame(predictions)
    print("Sample Predictions:")
    print(sub_df.head())

    # Save Submission
    save_submission(sub_df, Config.SUBMISSION_PATH)

    # Verify file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."
    loaded_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(loaded_sub) == len(sub_df), "Saved submission row count mismatch."
    assert list(loaded_sub.columns) == [
        "Patient_Week",
        "FVC",
        "Confidence",
    ], "Submission columns mismatch."

    print("Inference and submission verification successful.")

    # -------------------------------------------------------------------------
    # 7. Metric Utility Demo
    # -------------------------------------------------------------------------
    print("\n[7] Demonstrating Metric Calculation Utility...")

    # Create synthetic data
    # Case 1: Perfect prediction
    y_true = np.array([2000, 3000])
    y_pred = np.array([2000, 3000])
    sigma_pred = np.array([100, 100])  # Sigma > 70 (clipped)

    # Metric formula: - (sqrt(2) * Delta / sigma) - ln(sqrt(2) * sigma)
    # Delta = 0
    # Term 1 = 0
    # Term 2 = ln(sqrt(2) * 100) = ln(141.42) approx 4.95
    # Metric approx -4.95

    score = calculate_metric(y_true, y_pred, sigma_pred)
    print(f"Perfect Prediction Score (Sigma=100): {score:.4f}")

    # Expected calculation
    expected_score = -np.log(np.sqrt(2) * 100)
    assert np.isclose(
        score, expected_score, atol=1e-4
    ), "Metric calculation mismatch for perfect case"

    print("Metric utility verification successful.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
