import os
import sys
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import get_dataloaders
from library.model import (
    SLHDAN,
    train_one_epoch,
    validate,
    criterion,
    predict_and_submit,
)


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Configuration Overrides for Fast Baseline
    # Reducing epochs to ensure execution within 2 hours while allowing convergence
    Config.EPOCHS = 15
    Config.T_MAX = 15

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.BEST_MODEL_PATH), exist_ok=True)

    # 3. Data Loading
    # Uses cached data if available, otherwise processes it
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # 4. Model Initialization
    model = SLHDAN().to(device)

    # 5. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 6. Training Loop
    best_metric = -float("inf")

    # We will track the best metric to decide on submission
    print(f"Starting training for {Config.EPOCHS} epochs on {device}...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_metric = validate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        # Checkpointing
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 7. Final Evaluation and Failure Analysis
    # Load the best model for analysis
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    model.eval()

    # Variables for Failure Analysis
    all_targets = []
    all_preds = []
    all_sigmas = []
    all_tabular = []

    # Run inference on validation set to gather detailed data
    with torch.no_grad():
        for batch in val_loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            dt = batch["dt"].to(device)
            baseline = batch["baseline_fvc"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

            # Reconstruct predictions
            fvc_pred = baseline + alpha * dt
            sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

            # Store data
            all_targets.append(target.cpu().numpy())
            all_preds.append(fvc_pred.cpu().numpy())
            all_sigmas.append(sigma_pred.cpu().numpy())
            all_tabular.append(tabular.cpu().numpy())

    # Concatenate results
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    sigma = np.concatenate(all_sigmas)
    X_tab = np.concatenate(all_tabular)

    # Recalculate metric on full set to ensure precision matches requirement
    final_metric = laplace_log_likelihood_metric(y_true, y_pred, sigma)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation of Absolute Error with Features
    print("\n=== Failure Analysis ===")
    abs_error = np.abs(y_true - y_pred)

    # Feature names based on TabularPreprocessor in library/data.py
    # Indices: 0:Weeks, 1:Percent, 2:Age, 3:Sex, 4:Ex-smoker, 5:Never smoked, 6:Currently smokes
    feature_names = [
        "Weeks",
        "Percent",
        "Age",
        "Sex",
        "Ex-smoker",
        "Never smoked",
        "Currently smokes",
    ]

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(X_tab, columns=feature_names)
    analysis_df["AbsError"] = abs_error

    # Calculate correlations
    correlations = (
        analysis_df.corr()["AbsError"].drop("AbsError").sort_values(ascending=False)
    )
    print("Correlation between Input Features and Absolute Error:")
    print(correlations)

    # 8. Submission Logic
    # Threshold from requirements
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(test_loader)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
