import os
import sys
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.data import get_dataloaders
from library.model import CG_SDAN, predict
from library.train import run_training
from library.utils import seed_everything, compute_metric


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Limit epochs for a fast baseline execution
    Config.N_EPOCHS = 15

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"Initializing Fast Baseline Run (Epochs={Config.N_EPOCHS})...")

    # ==========================================
    # 2. Training Phase
    # ==========================================
    # run_training manages the loop, validation, and checkpoints
    run_training(debug=False)

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Failure Analysis...")

    # Load the best model saved during training
    device = torch.device(Config.DEVICE)
    model = CG_SDAN().to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Error: Model checkpoint not found.")
        return

    model.eval()

    # Get DataLoaders (re-fetching utilizes cache efficiently)
    _, val_loader, test_loader = get_dataloaders(debug=False)

    # Containers for analysis
    val_true_fvc = []
    val_pred_fvc = []
    val_pred_sigma = []

    # Containers for features (to correlate with error)
    # Tabular vector structure: [Week, Percent, Age, Sex, Smoke]
    feat_weeks = []
    feat_percent = []
    feat_age = []
    feat_sex = []
    feat_smoke = []

    # Inference Loop (No Grad)
    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)

            # Metadata for parametric reconstruction
            m_weeks = batch["meta"]["Weeks"].to(device).view(-1, 1)
            m_base_fvc = batch["meta"]["Baseline_FVC"].to(device).view(-1, 1)
            m_base_week = batch["meta"]["Baseline_Week"].to(device).view(-1, 1)

            # Forward pass
            alpha, s_base, s_growth = model(img_ax, img_cor, tabular)

            # Reconstruct predictions
            delta_t = m_weeks - m_base_week
            p_fvc = m_base_fvc + alpha * delta_t
            p_sigma = s_base + s_growth * torch.abs(delta_t)

            # Collect outputs
            val_true_fvc.extend(targets.cpu().numpy().flatten())
            val_pred_fvc.extend(p_fvc.cpu().numpy().flatten())
            val_pred_sigma.extend(p_sigma.cpu().numpy().flatten())

            # Collect features
            t_np = tabular.cpu().numpy()
            feat_weeks.extend(t_np[:, 0])
            feat_percent.extend(t_np[:, 1])
            feat_age.extend(t_np[:, 2])
            feat_sex.extend(t_np[:, 3])
            feat_smoke.extend(t_np[:, 4])

    # Compute Final Metric
    final_metric = compute_metric(val_true_fvc, val_pred_fvc, val_pred_sigma)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Features
    errors = np.abs(np.array(val_true_fvc) - np.array(val_pred_fvc))

    analysis_df = pd.DataFrame(
        {
            "Error": errors,
            "Week": feat_weeks,
            "Percent": feat_percent,
            "Age": feat_age,
            "Sex": feat_sex,
            "Smoke": feat_smoke,
        }
    )

    # Calculate correlation
    correlations = (
        analysis_df.corr()["Error"].drop("Error").sort_values(ascending=False)
    )
    print("\nCorrelation between Absolute Error and Input Features:")
    print(correlations)

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        # predict() handles loading the model, inference on test_loader, and saving to CSV
        predict(test_loader)
    else:
        print(
            f"\nMetric ({final_metric}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
