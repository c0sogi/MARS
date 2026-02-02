import os
import sys
import pandas as pd
import numpy as np
import torch
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss
from library.data import get_dataloaders
from library.model import HiFiDACR
from library.train import run_training, validate


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Training
    # Run training for a limited number of epochs to satisfy the "fast baseline" requirement
    # while ensuring sufficient learning.
    print("Starting training pipeline...")
    run_training(debug=False, num_epochs=12)

    # 3. Re-initialize Data and Model for Evaluation
    # We need to get the dataloaders again to access val/test and determine input dims
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # Determine tabular input dimension from a sample batch
    sample_batch = next(iter(train_loader))
    tab_dim = sample_batch["tabular"].shape[1]

    # Initialize model and load best weights
    model = HiFiDACR(tab_input_dim=tab_dim).to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print(f"Error: Best model not found at {Config.BEST_MODEL_PATH}")
        return

    model.eval()

    # 4. Validation & Metric Calculation
    criterion = LaplaceLogLikelihoodLoss().to(device)

    # validate() returns (avg_loss, metric_score)
    # The metric score is what we need.
    _, val_metric = validate(model, val_loader, criterion, device)

    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    print("Performing failure analysis on validation set...")

    # Load validation metadata to correlate errors with features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Collect predictions and targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tab = batch["tabular"].to(device)
            base_fvc = batch["baseline_fvc"].to(device)
            rel_week = batch["relative_week"].to(device)

            outputs = model(img_ax, img_cor, tab, base_fvc, rel_week)

            # Output col 0 is FVC
            all_preds.append(outputs[:, 0].cpu().numpy())
            all_targets.append(batch["target"].numpy())

    flat_preds = np.concatenate(all_preds)
    flat_targets = np.concatenate(all_targets).flatten()

    # Calculate absolute error
    errors = np.abs(flat_targets - flat_preds)

    # Ensure alignment (DataLoader should preserve order if shuffle=False)
    if len(errors) == len(val_df):
        val_df["Error"] = errors

        # Calculate correlations
        features_to_check = ["Age", "Percent", "Weeks"]
        for feat in features_to_check:
            if feat in val_df.columns:
                corr = val_df[feat].corr(val_df["Error"])
                print(f"Error Correlation with {feat}: {corr:.6f}")
    else:
        print(
            f"Warning: Mismatch between validation predictions ({len(errors)}) and metadata rows ({len(val_df)}). Skipping correlation analysis."
        )

    # 6. Submission Generation
    threshold = -6.510164260864258

    if val_metric > threshold:
        print(
            f"Validation metric {val_metric} exceeds threshold {threshold}. Generating submission..."
        )

        submission_rows = []

        with torch.no_grad():
            for batch in test_loader:
                img_ax = batch["image_axial"].to(device)
                img_cor = batch["image_coronal"].to(device)
                tab = batch["tabular"].to(device)
                base_fvc = batch["baseline_fvc"].to(device)
                rel_week = batch["relative_week"].to(device)
                patient_weeks = batch["patient_week"]

                outputs = model(img_ax, img_cor, tab, base_fvc, rel_week)

                fvc_preds = outputs[:, 0].cpu().numpy()
                conf_preds = outputs[:, 1].cpu().numpy()

                for pw, f, c in zip(patient_weeks, fvc_preds, conf_preds):
                    submission_rows.append(
                        {"Patient_Week": pw, "FVC": f, "Confidence": c}
                    )

        # Create DataFrame and save
        sub_df = pd.DataFrame(submission_rows)

        # Ensure correct column order
        sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]

        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric {val_metric} does not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
