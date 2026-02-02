import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.data import get_dataloaders
from library.model import MultiAxisTriSlabModel
from library.train import (
    train_one_epoch,
    validate,
    generate_submission,
    ParametricLoss,
    predict_trajectory,
)

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    # We reduce epochs to ensure it finishes quickly within the limit
    Config.EPOCHS = 12

    print(f"Running Fast Baseline with {Config.EPOCHS} epochs...")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model Initialization
    print("Initializing Model...")
    model = MultiAxisTriSlabModel(
        backbone_name="efficientnet_b0",
        pretrained=True,
        tabular_input_dim=5,  # Age, Sex, 3xSmoking
        output_dim=3,  # alpha, sigma_base, sigma_growth
    )
    model = model.to(Config.DEVICE)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = ParametricLoss()

    # 5. Training Loop
    best_metric = -float("inf")

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_metric = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, Config.DEVICE
        )

        # Validate
        val_metric = validate(model, val_loader, criterion, Config.DEVICE)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Train: {train_metric:.5f} | Val: {val_metric:.5f}"
        )

        # Save Best Model
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 6. Final Evaluation & Failure Analysis
    print("\nRunning Final Evaluation on Best Model...")

    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )

    model.eval()

    # Storage for analysis
    all_targets = []
    all_pred_fvc = []
    all_pred_sigma = []
    all_errors = []

    # Features for correlation: [Time_Delta, Age, Sex]
    # Note: Smoking is one-hot, we'll skip complex correlation for it to keep it simple or take argmax
    feature_data = {"Time_Delta": [], "Age_Norm": [], "Sex": []}

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            axial = batch["axial"].to(Config.DEVICE)
            coronal = batch["coronal"].to(Config.DEVICE)
            sagittal = batch["sagittal"].to(Config.DEVICE)
            tabular = batch["tabular"].to(Config.DEVICE)
            targets = batch["target"].to(Config.DEVICE)
            base_fvc = batch["base_fvc"].to(Config.DEVICE)
            time_delta = batch["time_delta"].to(Config.DEVICE)

            # Inference
            outputs = model(axial, coronal, sagittal, tabular)
            pred_fvc, pred_sigma = predict_trajectory(outputs, base_fvc, time_delta)

            # Collect Metric Data
            all_targets.append(targets.cpu().numpy())
            all_pred_fvc.append(pred_fvc.cpu().numpy())
            all_pred_sigma.append(pred_sigma.cpu().numpy())

            # Calculate Error for Analysis
            abs_error = torch.abs(targets - pred_fvc).cpu().numpy()
            all_errors.append(abs_error)

            # Collect Features
            feature_data["Time_Delta"].append(time_delta.cpu().numpy())
            feature_data["Age_Norm"].append(tabular[:, 0].cpu().numpy())
            feature_data["Sex"].append(tabular[:, 1].cpu().numpy())

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_pred_fvc = np.concatenate(all_pred_fvc)
    all_pred_sigma = np.concatenate(all_pred_sigma)
    all_errors = np.concatenate(all_errors)

    for k in feature_data:
        feature_data[k] = np.concatenate(feature_data[k])

    # Compute Final Metric
    # We re-compute using the utility to ensure exact match with requirements
    final_metric = laplace_log_likelihood(all_targets, all_pred_fvc, all_pred_sigma)

    # REQUIRED OUTPUT: Final Validation Metric
    # Printing full precision as requested
    print(f"Final Validation Metric: {final_metric.item()}")

    # Failure Analysis: Correlation
    print("\nFailure Analysis (Correlation with Absolute Error):")
    df_analysis = pd.DataFrame(
        {
            "Error": all_errors,
            "Time_Delta": feature_data["Time_Delta"],
            "Age_Norm": feature_data["Age_Norm"],
            "Sex": feature_data["Sex"],
        }
    )

    correlations = df_analysis.corr()["Error"].drop("Error")
    print(correlations)

    # 7. Submission
    THRESHOLD = -6.569469928741455

    if final_metric.item() > THRESHOLD:
        print(
            f"\nMetric ({final_metric.item()}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, Config.DEVICE)
    else:
        print(
            f"\nMetric ({final_metric.item()}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
