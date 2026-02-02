import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(".")

# Import library components
from library.config import Config
from library.data import get_dataloaders, prepare_dataframe
from library.model import ASADAN, laplace_log_likelihood_loss
from library.train import run_training
from library.utils import seed_everything, score


def main():
    print("=== Starting Demonstration Script ===")

    # ==========================================
    # 1. Configure for Fast Demonstration
    # ==========================================
    print("\n[1] Configuring parameters for demo...")
    # Modify Config class attributes globally for this run
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SAMPLE_SIZE = 16  # Use a tiny subset
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.PATIENCE = 2  # Disable early stopping effectively for 2 epochs

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # ==========================================
    # 2. Verify Data Pipeline
    # ==========================================
    print("\n[2] Verifying Data Pipeline...")

    # Get dataloaders in debug mode
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,  # Will try to load cache or generate on fly
        debug=True,
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    # Extract components
    img_axial = batch["img_axial"]
    img_coronal = batch["img_coronal"]
    static_features = batch["static_features"]
    baseline_fvc = batch["baseline_fvc"]
    week = batch["week"]
    target = batch["target"]

    # Validate Shapes
    # Image: (B, 3, 224, 224)
    assert img_axial.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Axial Image Shape: {img_axial.shape}"
    assert img_coronal.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Coronal Image Shape: {img_coronal.shape}"
    # Static Features: (B, 4) -> [Age, Sex, Smoking, Percent]
    assert static_features.shape == (
        Config.BATCH_SIZE,
        4,
    ), f"Incorrect Static Features Shape: {static_features.shape}"
    # Target: (B,)
    assert target.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect Target Shape: {target.shape}"

    print("    Data shapes verified successfully.")

    # ==========================================
    # 3. Verify Model Architecture & Forward Pass
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    model = ASADAN().to(device)

    # Move batch to device
    img_axial = img_axial.to(device)
    img_coronal = img_coronal.to(device)
    static_features = static_features.to(device)
    baseline_fvc = baseline_fvc.to(device)
    week = week.to(device)
    target = target.to(device)

    # Forward Pass
    model.eval()
    with torch.no_grad():
        fvc_pred, sigma_pred = model(
            img_axial, img_coronal, static_features, baseline_fvc, week
        )

    # Validate Outputs
    assert fvc_pred.shape == (Config.BATCH_SIZE,), "Prediction shape mismatch"
    assert sigma_pred.shape == (Config.BATCH_SIZE,), "Confidence shape mismatch"

    # Validate Constraints (Sigma must be positive)
    if (sigma_pred <= 0).any():
        raise ValueError("Model produced non-positive confidence values!")

    print(f"    Forward pass successful.")
    print(f"    Sample Preds: {fvc_pred[:2].cpu().numpy()}")
    print(f"    Sample Sigma: {sigma_pred[:2].cpu().numpy()}")

    # Verify Loss Function
    loss = laplace_log_likelihood_loss(target, fvc_pred, sigma_pred)
    assert not torch.isnan(loss), "Loss returned NaN"
    print(f"    Sample Loss: {loss.item():.4f}")

    # ==========================================
    # 4. Run Training Loop
    # ==========================================
    print("\n[4] Running Training Simulation...")

    # This calls the library function which handles the loop, validation, and saving
    best_model_path = run_training(debug=True)

    assert os.path.exists(best_model_path), "Best model file was not saved!"
    print(f"    Training complete. Model saved at: {best_model_path}")

    # ==========================================
    # 5. Inference & Submission Generation
    # ==========================================
    print("\n[5] Generating Submission...")

    # Load the best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Prepare to collect predictions
    predictions = []

    # We need to map predictions back to Patient_Week IDs.
    # In debug mode, the loader uses the first N rows of the dataframe.
    # We load the dataframe manually to get the IDs corresponding to the loader.
    test_df = prepare_dataframe(Config.TEST_CSV, mode="test")
    if Config.DEBUG_SAMPLE_SIZE:
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    print(f"    Inference on {len(test_df)} samples...")

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # Move to device
            img_axial = data["img_axial"].to(device)
            img_coronal = data["img_coronal"].to(device)
            static_features = data["static_features"].to(device)
            baseline_fvc = data["baseline_fvc"].to(device)
            week = data["week"].to(device)

            # Predict
            fvc_pred, sigma_pred = model(
                img_axial, img_coronal, static_features, baseline_fvc, week
            )

            # Store results
            batch_size = img_axial.size(0)
            for i in range(batch_size):
                predictions.append(
                    {"FVC": fvc_pred[i].item(), "Confidence": sigma_pred[i].item()}
                )

    # Create Submission DataFrame
    # Note: test_loader preserves order of test_df
    pred_df = pd.DataFrame(predictions)
    submission = pd.concat([test_df.reset_index(drop=True), pred_df], axis=1)

    # Keep only required columns
    submission = submission[["Patient_Week", "FVC", "Confidence"]]

    # Verify format
    assert "Patient_Week" in submission.columns
    assert "FVC" in submission.columns
    assert "Confidence" in submission.columns
    assert len(submission) == len(test_df)

    # Save
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"    Submission saved to: {submission_path}")
    print(submission.head())

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
