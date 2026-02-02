import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything, calculate_metric, AverageMeter
from library.dataset import LungDataset
from library.model import GranularTabularNetwork
from library.loss import RobustLaplaceLogLikelihoodLoss
from library.training import train_one_epoch, validate_one_epoch


def analyze_failures(val_loader, model, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between error and features.
    """
    model.eval()

    data_records = []

    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            age = batch["age"].to(device)
            sex = batch["sex"].to(device)
            smoke = batch["smoke"].to(device)
            percent = batch["percent"].to(device)
            priors = batch["priors"].to(device)
            time_delta = batch["time_delta"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            fvc_pred, conf_pred = model(
                axial, coronal, age, sex, smoke, percent, priors, time_delta
            )

            # Move to CPU for analysis
            fvc_pred_np = fvc_pred.cpu().numpy()
            target_np = target.cpu().numpy()

            # Extract features for correlation (un-normalize where possible or use raw)
            # Note: We use the normalized values passed to model for correlation as proxies
            age_np = age.cpu().numpy()
            percent_np = percent.cpu().numpy()
            time_delta_np = time_delta.cpu().numpy()
            sex_np = sex.cpu().numpy()
            smoke_np = smoke.cpu().numpy()

            for i in range(len(fvc_pred_np)):
                error = np.abs(fvc_pred_np[i] - target_np[i])
                data_records.append(
                    {
                        "Error": error,
                        "Age_Norm": age_np[i],
                        "Percent_Norm": percent_np[i],
                        "Time_Delta": time_delta_np[i],
                        "Sex_Enc": sex_np[i],
                        "Smoke_Enc": smoke_np[i],
                    }
                )

    df_analysis = pd.DataFrame(data_records)

    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)
    print(f"Analyzed {len(df_analysis)} validation samples.")

    # Calculate correlations
    if not df_analysis.empty:
        correlations = df_analysis.corr()["Error"].sort_values(ascending=False)
        print("\nCorrelation between Absolute Error and Features:")
        print(correlations)
    else:
        print("No data for analysis.")


def generate_test_submission(model, device):
    """
    Generates submission file for the test set.
    """
    print("\nGenerating submission for test set...")
    model.eval()

    test_dataset = LungDataset(mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
    )

    results = []

    with torch.no_grad():
        for batch in test_loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            age = batch["age"].to(device)
            sex = batch["sex"].to(device)
            smoke = batch["smoke"].to(device)
            percent = batch["percent"].to(device)
            priors = batch["priors"].to(device)
            time_delta = batch["time_delta"].to(device)
            patient_weeks = batch["patient_week"]

            fvc_pred, conf_pred = model(
                axial, coronal, age, sex, smoke, percent, priors, time_delta
            )

            fvc_pred = fvc_pred.cpu().numpy()
            conf_pred = conf_pred.cpu().numpy()

            for pw, fvc, conf in zip(patient_weeks, fvc_pred, conf_pred):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    submission_df = pd.DataFrame(results)
    submission_df = submission_df[["Patient_Week", "FVC", "Confidence"]]
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def main():
    # 1. Setup
    # Override Config for Fast Baseline
    Config.N_EPOCHS = 20  # Limit epochs for speed

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = LungDataset(mode="train")
    val_dataset = LungDataset(mode="val")

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

    # 3. Model & Optimization
    print("Initializing Model...")
    model = GranularTabularNetwork()
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.N_EPOCHS, eta_min=Config.ETA_MIN
    )

    loss_fn = RobustLaplaceLogLikelihoodLoss()

    # 4. Training Loop
    best_score = -float("inf")

    print(f"Starting training for {Config.N_EPOCHS} epochs...")
    for epoch in range(Config.N_EPOCHS):
        # Train
        train_loss, train_score = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scheduler
        )

        # Validate
        val_loss, val_score = validate_one_epoch(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch+1}/{Config.N_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Score: {val_score:.4f}"
        )

        # Save Best
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print("Training complete.")

    # 5. Final Validation & Failure Analysis
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Re-run validation to get exact final metric and perform analysis
    final_loss, final_metric = validate_one_epoch(model, val_loader, loss_fn, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    analyze_failures(val_loader, model, device)

    # 6. Conditional Submission
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission."
        )
        generate_test_submission(model, device)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
