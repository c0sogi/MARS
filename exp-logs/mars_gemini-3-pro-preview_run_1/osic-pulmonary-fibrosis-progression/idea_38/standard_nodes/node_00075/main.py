import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data import LungDataset
from library.model import DAVRNet
from library.engine import fit, evaluate, predict


def main():
    # 1. Setup and Configuration
    # Override epochs for a fast baseline execution
    Config.epochs = 25

    seed_everything(Config.seed)
    device = torch.device(Config.device)

    print(f"Initializing run on {device} with {Config.epochs} epochs...")

    # 2. Data Loading
    # Load datasets with caching enabled
    train_dataset = LungDataset(mode="train")
    val_dataset = LungDataset(mode="val")
    test_dataset = LungDataset(mode="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    # 3. Model Initialization
    model = DAVRNet().to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.eta_min
    )

    # 4. Training
    fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        Config.epochs,
        Config.patience,
    )

    # 5. Evaluation
    # Load the best model checkpoint
    model.load_state_dict(torch.load(Config.model_path, map_location=device))

    # Compute final metric on the full validation set
    final_metric = evaluate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis...")
    model.eval()

    errors = []
    features = []

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            rel_week = batch["rel_week"].to(device).unsqueeze(1)
            true_fvc = batch["fvc"].to(device).unsqueeze(1)
            baseline_fvc = batch["baseline_fvc"].to(device).unsqueeze(1)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

            # Reconstruct predictions
            pred_fvc = baseline_fvc + alpha * rel_week

            # Calculate absolute error
            batch_errors = torch.abs(true_fvc - pred_fvc).cpu().numpy().flatten()
            errors.extend(batch_errors)

            # Extract features for correlation
            # Tabular structure: [Age_Norm, Sex(2), Smoke(3), Percent_Norm]
            # Index 0: Age, Index -1: Percent
            tabs = tabular.cpu().numpy()
            weeks = rel_week.cpu().numpy().flatten()
            base_fvc = baseline_fvc.cpu().numpy().flatten()

            for i in range(len(batch_errors)):
                features.append(
                    {
                        "Age_Norm": tabs[i, 0],
                        "Percent_Norm": tabs[i, -1],
                        "Rel_Week": weeks[i],
                        "Baseline_FVC": base_fvc[i],
                    }
                )

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(features)
    df_analysis["Error_Magnitude"] = errors

    # Calculate correlations
    correlations = df_analysis.corr()["Error_Magnitude"].drop("Error_Magnitude")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 7. Submission
    threshold = -6.510164260864258
    if final_metric > threshold:
        print(
            f"\nMetric {final_metric} exceeds threshold {threshold}. Generating submission..."
        )
        predict(model, test_loader, device)
    else:
        print(
            f"\nMetric {final_metric} does not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
