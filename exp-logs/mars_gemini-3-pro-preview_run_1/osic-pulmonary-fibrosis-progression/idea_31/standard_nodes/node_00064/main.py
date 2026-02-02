import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Ensure library modules are accessible
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, metric_laplace_log_likelihood
from library.data import OSICDataset
from library.model import AVRDAN
from library.train import run_training
from library.predict import generate_submission_file

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline execution
    # Reducing epochs to ensure completion within 2 hours while allowing convergence
    Config.EPOCHS = 25

    print(f"=== Starting AVR-DAN Pipeline on {Config.DEVICE} ===")

    # 2. Training
    # This will train the model and save the best weights to Config.MODEL_SAVE_PATH
    run_training()

    # 3. Validation & Failure Analysis
    print("\n=== Starting Validation & Failure Analysis ===")
    device = Config.DEVICE

    # Load Validation Data
    val_dataset = OSICDataset(csv_path=Config.VAL_CSV, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model = AVRDAN()
    model.to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("CRITICAL ERROR: Model weights not found. Training may have failed.")
        return

    model.eval()

    # Containers for validation results
    all_targets = []
    all_fvc_preds = []
    all_sigma_preds = []
    all_metas_skip = []  # Contains [BaseFVC_scaled, BasePct_scaled, Age_norm]

    # Inference Loop
    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tab_glu = batch["tab_glu"].to(device)
            tab_skip = batch["tab_skip"].to(device)
            delta_week = batch["delta_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            fvc_pred, conf_pred = model(
                img_ax, img_cor, tab_glu, tab_skip, delta_week, baseline_fvc
            )

            # Collect results (move to CPU)
            all_targets.append(target.cpu())
            all_fvc_preds.append(fvc_pred.cpu())
            all_sigma_preds.append(conf_pred.cpu())
            all_metas_skip.append(tab_skip.cpu())

    # Concatenate all batches
    y_true = torch.cat(all_targets)
    y_pred = torch.cat(all_fvc_preds)
    sigma = torch.cat(all_sigma_preds)
    metas_skip = torch.cat(all_metas_skip)

    # Compute Final Metric
    # metric_laplace_log_likelihood returns a tensor, use .item() for scalar
    final_metric = metric_laplace_log_likelihood(y_true, y_pred, sigma).item()

    # Print strictly formatted metric
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Calculate absolute error
    errors = torch.abs(y_true - y_pred).numpy()

    # Reconstruct features from normalized tensors for meaningful correlation
    # tab_skip structure: [Baseline_FVC / 1000.0, Baseline_Percent / 100.0, Age_Norm]
    # Age_Norm = (Age - 50) / 20.0

    baseline_fvc_vals = metas_skip[:, 0].numpy() * 1000.0
    baseline_pct_vals = metas_skip[:, 1].numpy() * 100.0
    age_vals = metas_skip[:, 2].numpy() * 20.0 + 50.0

    analysis_df = pd.DataFrame(
        {
            "Abs_Error": errors,
            "Baseline_FVC": baseline_fvc_vals,
            "Baseline_Percent": baseline_pct_vals,
            "Age": age_vals,
        }
    )

    # Compute correlations
    correlations = analysis_df.corr()["Abs_Error"].drop("Abs_Error")

    print("\nFailure Analysis - Correlation with Absolute Error:")
    print(correlations)

    # 4. Submission Generation
    # Threshold defined in task
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission_file()
    else:
        print(
            f"\nMetric ({final_metric}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
