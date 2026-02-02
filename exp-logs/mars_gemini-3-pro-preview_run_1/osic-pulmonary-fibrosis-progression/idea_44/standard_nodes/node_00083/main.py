import os
import sys
import torch
import numpy as np
import pandas as pd
from library.utils import seed_everything, score_function
from library.engine import Engine


def main():
    # 1. Setup and Configuration
    # Ensure reproducibility
    seed_everything(42)

    # Initialize the Engine
    # We use 15 epochs to ensure the training completes quickly (Fast Baseline)
    # while allowing the model to learn the shared representation.
    engine = Engine(debug=False, epochs=15)

    # 2. Model Training
    print("Starting training pipeline...")
    engine.fit()

    # 3. Validation and Metric Calculation
    print("Performing final validation...")

    # Load the best model checkpoint
    best_path = os.path.join(engine.cfg.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_path):
        engine.model.load_state_dict(torch.load(best_path, map_location=engine.device))
    else:
        print("Warning: Best model checkpoint not found. Using current weights.")

    engine.model.eval()

    # Containers for analysis
    all_targets = []
    all_preds = []
    all_sigmas = []
    all_tabular = []

    # Inference loop on Validation Set
    val_loader = engine.val_loader
    device = engine.device

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tab = batch["tabular"].to(device)
            target = batch["target"].to(device)
            week = batch["week"].to(device)
            pids = batch["patient_id"]

            # Get baseline info
            base_fvc, base_week = engine._get_baseline_batch(pids)

            # Forward pass
            preds = engine.model(img_ax, img_cor, tab)
            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            # Reconstruct Predictions based on Anchored Trajectory Logic
            dt = week - base_week
            pred_fvc = base_fvc + alpha * dt
            pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

            # Collect results (move to CPU)
            all_targets.append(target.cpu())
            all_preds.append(pred_fvc.cpu())
            all_sigmas.append(pred_sigma.cpu())
            all_tabular.append(tab.cpu())

    # Concatenate all batches
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    sigma = torch.cat(all_sigmas).numpy()
    tabular = torch.cat(all_tabular).numpy()

    # Calculate Final Metric
    final_metric = score_function(y_true, y_pred, sigma)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming failure analysis...")

    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Extract features from normalized tabular tensor
    # Tensor layout from LungDataset:
    # [Age, Percent, Sex0(M), Sex1(F), Smoke0(Ex), Smoke1(Never), Smoke2(Curr), Pad, Pad]
    age = tabular[:, 0]
    percent = tabular[:, 1]
    sex_female = tabular[:, 3]
    smoke_never = tabular[:, 5]
    smoke_current = tabular[:, 6]

    # Create DataFrame for correlation analysis
    analysis_df = pd.DataFrame(
        {
            "Error": errors,
            "Age": age,
            "Percent": percent,
            "Sex_Female": sex_female,
            "Smoke_Never": smoke_never,
            "Smoke_Current": smoke_current,
        }
    )

    # Compute correlations
    correlations = analysis_df.corr()["Error"].sort_values(ascending=False)
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 5. Submission Generation
    # Threshold defined in task
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        engine.generate_submission()
    else:
        print(
            f"\nMetric ({final_metric}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
