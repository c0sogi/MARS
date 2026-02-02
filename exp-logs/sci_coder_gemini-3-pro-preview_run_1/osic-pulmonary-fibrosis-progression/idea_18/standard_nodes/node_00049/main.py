import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config, setup_reproducibility
from library.data import LungDataset, get_transforms, prepare_data
from library.model import CalibratedSymmetricDualAxisNetwork
from library.train import train_epoch, validate_epoch, LaplaceLoss
from library.utils import calculate_metric


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between error magnitude and input features.
    """
    model.eval()
    results = []

    print("\nRunning Failure Analysis on Validation Set...")

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            imgs_ax = batch["image_axial"].to(device)
            imgs_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)

            # Metadata for reconstruction
            weeks = batch["metadata"]["Weeks"].to(device)
            base_weeks = batch["metadata"]["Baseline_Week"].to(device)
            base_fvc = batch["metadata"]["Baseline_FVC"].to(device)

            # Forward Pass
            alpha, sigma_base, sigma_growth = model(imgs_ax, imgs_cor, tabular)

            # Reconstruct Predictions
            dt = weeks - base_weeks
            y_pred = base_fvc + alpha.view(-1) * dt

            # Calculate Error
            abs_error = torch.abs(targets - y_pred).cpu().numpy()

            # Extract features for correlation
            # Tabular is [Age_norm, Sex_M, Sex_F, Smoke_Ex, Smoke_Never, Smoke_Curr, Percent_norm]
            # We want to map back to roughly interpretable values or just use the norms
            tab_np = tabular.cpu().numpy()

            batch_size = len(targets)
            for i in range(batch_size):
                results.append(
                    {
                        "Abs_Error": abs_error[i],
                        "Weeks": weeks[i].item(),
                        "Age_Norm": tab_np[i, 0],
                        "Percent_Norm": tab_np[i, 6],
                        "Is_Male": tab_np[i, 1],
                        "Is_Current_Smoker": tab_np[i, 5],
                    }
                )

    df_results = pd.DataFrame(results)

    # Calculate correlations
    correlations = df_results.corr()["Abs_Error"].sort_values(ascending=False)
    print("\nCorrelation between Absolute Error and Features:")
    print(correlations.drop("Abs_Error"))

    return df_results


def generate_submission(model, test_df, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("\nGenerating Submission...")

    # Prepare Test Dataset
    test_dataset = LungDataset(
        test_df,
        cache_dir=Config.CACHE_DIR,
        transform=get_transforms("val"),  # Deterministic transforms
        mode="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            imgs_ax = batch["image_axial"].to(device)
            imgs_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)

            # Metadata
            patient_weeks = batch["metadata"]["Patient_Week"]
            weeks = batch["metadata"]["Weeks"].to(device)
            base_weeks = batch["metadata"]["Baseline_Week"].to(device)
            base_fvc = batch["metadata"]["Baseline_FVC"].to(device)

            # Forward Pass
            alpha, sigma_base, sigma_growth = model(imgs_ax, imgs_cor, tabular)

            # Calculate Predictions
            dt = weeks - base_weeks

            # FVC = Baseline + alpha * dt
            pred_fvc = base_fvc + alpha.view(-1) * dt

            # Confidence = sigma_base + sigma_growth * |dt|
            pred_sigma = sigma_base.view(-1) + sigma_growth.view(-1) * torch.abs(dt)

            # Clip Confidence
            pred_sigma = torch.clamp(pred_sigma, min=Config.CONFIDENCE_CLIP)

            # Collect results
            pred_fvc_np = pred_fvc.cpu().numpy()
            pred_sigma_np = pred_sigma.cpu().numpy()

            for i in range(len(patient_weeks)):
                predictions.append(
                    {
                        "Patient_Week": patient_weeks[i],
                        "FVC": pred_fvc_np[i],
                        "Confidence": pred_sigma_np[i],
                    }
                )

    # Create DataFrame
    sub_df = pd.DataFrame(predictions)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH} with {len(sub_df)} rows.")


def main():
    # 1. Setup & Configuration
    setup_reproducibility(Config.SEED)
    Config.setup_directories()

    # Adjust Config for Fast Baseline
    Config.EPOCHS = 15
    Config.PATIENCE = 5
    Config.BATCH_SIZE = 32

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Load Data
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)

    # Initialize Datasets
    train_dataset = LungDataset(
        train_df,
        cache_dir=Config.CACHE_DIR,
        transform=get_transforms("train"),
        mode="train",
    )

    val_dataset = LungDataset(
        val_df, cache_dir=Config.CACHE_DIR, transform=get_transforms("val"), mode="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = CalibratedSymmetricDualAxisNetwork().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLoss()

    # 4. Training Loop
    best_score = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate_epoch(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Metric: {val_score:.6f} | Time: {elapsed:.1f}s"
        )

        # Early Stopping
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best Score! Saved.")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Calculate Final Metric on full validation set
    final_metric = validate_epoch(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    analyze_failures(model, val_loader, device)

    # 6. Submission Logic
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        try:
            test_df = pd.read_csv(Config.TEST_META_PATH)
            generate_submission(model, test_df, device)
        except Exception as e:
            print(f"Error generating submission: {e}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
