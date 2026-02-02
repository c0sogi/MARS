import os
import sys
import torch
import numpy as np
import pandas as pd
from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import CVERNet, predict_and_submit
from library.train import train_one_epoch, validate, LaplaceLikelihoodLoss
from library.utils import score_function


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    # We use debug=False to train on the full dataset for best performance.
    # The caching mechanism in library.data will speed up image loading.
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # 3. Model Initialization
    model = CVERNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )
    loss_fn = LaplaceLikelihoodLoss()

    # 4. Training Loop
    best_metric = -float("inf")
    patience_counter = 0
    best_model_path = Config.BEST_MODEL_PATH

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Score: {val_score:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_score > best_metric:
            best_metric = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print("Training complete.")

    # 5. Final Evaluation & Failure Analysis
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Collect data for analysis
    val_preds = []
    val_sigmas = []
    val_targets = []
    val_weeks = []
    val_base_fvc = []
    val_tabular_features = []  # To store (Age, Percent, etc.)

    with torch.no_grad():
        for batch in val_loader:
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            base_week = batch["base_week"].to(device)

            fvc_pred, sigma_pred = model(
                img_ax, img_cor, tabular, weeks, base_fvc, base_week
            )

            val_preds.extend(fvc_pred.cpu().numpy())
            val_sigmas.extend(sigma_pred.cpu().numpy())
            val_targets.extend(target.cpu().numpy())
            val_weeks.extend(weeks.cpu().numpy())
            val_base_fvc.extend(base_fvc.cpu().numpy())
            val_tabular_features.append(tabular.cpu().numpy())

    # Convert to numpy
    y_true = np.array(val_targets)
    y_pred = np.array(val_preds)
    sigma = np.array(val_sigmas)
    weeks_arr = np.array(val_weeks)
    base_fvc_arr = np.array(val_base_fvc)
    tabular_arr = np.concatenate(val_tabular_features, axis=0)

    # Compute Final Metric
    final_metric = score_function(y_true, y_pred, sigma)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis
    # Calculate absolute error
    abs_error = np.abs(y_true - y_pred)

    # Construct DataFrame for correlation analysis
    # Tabular features: Index 0 is Age (scaled), Index 1 is Percent (scaled)
    # We use the scaled versions which is sufficient for correlation direction/magnitude
    analysis_df = pd.DataFrame(
        {
            "Error": abs_error,
            "Weeks": weeks_arr,
            "Base_FVC": base_fvc_arr,
            "Age_Scaled": tabular_arr[:, 0],
            "Percent_Scaled": tabular_arr[:, 1],
        }
    )

    print("\nFailure Analysis (Correlation with Absolute Error):")
    correlations = analysis_df.corr()["Error"].sort_values(ascending=False)
    print(correlations)

    # 6. Submission
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.5f}) is better than threshold ({THRESHOLD:.5f}). Generating submission..."
        )
        # predict_and_submit handles loading the best model internally
        predict_and_submit(test_loader)
    else:
        print(
            f"\nMetric ({final_metric:.5f}) did not meet threshold ({THRESHOLD:.5f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
