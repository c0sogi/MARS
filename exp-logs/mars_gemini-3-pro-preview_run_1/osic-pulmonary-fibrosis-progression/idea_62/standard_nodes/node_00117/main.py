import os
import sys
import importlib
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Cite debug_lesson_7: Stale Module Cache Masks Fixes in Persistent Runtimes
import library.data
import library.model
import library.train

importlib.reload(library.data)
importlib.reload(library.model)
importlib.reload(library.train)

from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.data import get_dataloaders
from library.model import NDSSLN
from library.train import train_epoch, validate, generate_submission


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Initializing run on {device}...")

    # 2. Data Loading
    # We use debug=False to train on the full dataset for the best possible score.
    # The dataset size is small enough that this will complete quickly.
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # 3. Model Initialization
    model = NDSSLN().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 4. Training Loop
    best_score = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss:.4f} - Val Score: {val_score:.6f}"
        )

        # Checkpoint & Early Stopping
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! Score: {best_score:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break

    # 5. Final Evaluation & Failure Analysis
    print("\nLoading best model for analysis...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model found. Using current model state.")

    model.eval()

    # Collectors for analysis
    all_true = []
    all_pred = []
    all_sigma = []
    all_tabular = []

    with torch.no_grad():
        for batch in val_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)

            base_fvc = batch["base_fvc"].to(device)
            delta_week = batch["delta_week"].to(device)
            target_fvc = batch["target"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Calculate Predictions
            fvc_pred = base_fvc + alpha * delta_week
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

            # Collect data
            all_true.append(target_fvc.cpu().numpy())
            all_pred.append(fvc_pred.cpu().numpy())
            all_sigma.append(sigma_pred.cpu().numpy())
            all_tabular.append(tabular.cpu().numpy())

    # Concatenate results
    y_true = np.concatenate(all_true).flatten()
    y_pred = np.concatenate(all_pred).flatten()
    sigma = np.concatenate(all_sigma).flatten()
    tabular_data = np.concatenate(all_tabular, axis=0)

    # Compute Final Metric
    final_metric = laplace_log_likelihood(y_true, y_pred, sigma)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Input Features and Error Magnitude
    print("\n=== Failure Analysis ===")
    errors = np.abs(y_true - y_pred)

    # Feature names corresponding to LungDataset construction
    # [age_norm, sex_bin, s_ex, s_never, s_curr, percent_norm]
    feature_names = [
        "Age",
        "Sex_Female",
        "Smoke_Ex",
        "Smoke_Never",
        "Smoke_Curr",
        "Percent",
    ]

    analysis_df = pd.DataFrame(tabular_data, columns=feature_names)
    analysis_df["Error_Magnitude"] = errors

    # Compute correlation
    correlations = analysis_df.corr()["Error_Magnitude"].drop("Error_Magnitude")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations.sort_values(ascending=False))

    # 6. Conditional Submission
    threshold = -6.510164260864258

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
