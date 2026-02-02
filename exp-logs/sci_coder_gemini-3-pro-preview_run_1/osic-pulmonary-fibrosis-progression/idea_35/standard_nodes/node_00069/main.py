import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import OSICDataset, get_transforms
from library.model import LARFNet
from library.train import run_training


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Training Phase
    # Execute the training pipeline provided in library/train.py
    # This will train the model and save the best weights to Config.CACHE_DIR/best_model.pth
    print("=== Starting Training Phase ===")
    run_training()

    # 3. Validation & Failure Analysis
    print("\n=== Starting Validation & Failure Analysis ===")

    # Load the best model
    model = LARFNet().to(device)
    checkpoint_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"CRITICAL ERROR: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading best model from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Prepare Validation Data
    val_dataset = OSICDataset(
        csv_path=Config.VAL_CSV, mode="val", transform=get_transforms(mode="val")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Validation Inference
    y_true = []
    y_pred = []
    y_sigma = []

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            img_axial = batch["image_axial"].to(device)
            img_coronal = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            week = batch["week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)

            # Forward pass
            fvc, sigma = model(
                img_axial, img_coronal, tabular, week=week, baseline_fvc=baseline_fvc
            )

            # Collect results
            y_true.append(batch["target"].numpy())
            y_pred.append(fvc.cpu().numpy())
            y_sigma.append(sigma.cpu().numpy())

    # Concatenate results
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_sigma = np.concatenate(y_sigma)

    # Calculate and Print Metric
    final_metric = calculate_metric(y_true, y_pred, y_sigma)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis (Correlation with Absolute Error) ---")
    val_df = val_dataset.df
    errors = np.abs(y_true - y_pred)

    # Features to analyze
    analysis_features = ["Age", "Percent", "Weeks", "Baseline_FVC"]

    for feat in analysis_features:
        if feat in val_df.columns:
            vals = val_df[feat].values
            # Calculate Pearson correlation using numpy
            # Handle potential NaNs just in case, though data is clean
            valid_mask = ~np.isnan(vals) & ~np.isnan(errors)
            if np.sum(valid_mask) > 1:
                corr = np.corrcoef(vals[valid_mask], errors[valid_mask])[0, 1]
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Insufficient data for correlation")
        else:
            print(f"  {feat}: Feature not found in dataframe")

    # 4. Submission Generation
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\n=== Generating Submission (Metric {final_metric:.4f} > {THRESHOLD:.4f}) ==="
        )

        # Prepare Test Data
        test_dataset = OSICDataset(
            csv_path=Config.TEST_CSV,
            mode="test",
            transform=get_transforms(mode="val"),  # No augmentation for inference
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        submission_rows = []

        with torch.no_grad():
            for batch in test_loader:
                img_axial = batch["image_axial"].to(device)
                img_coronal = batch["image_coronal"].to(device)
                tabular = batch["tabular"].to(device)
                week = batch["week"].to(device)
                baseline_fvc = batch["baseline_fvc"].to(device)
                patient_weeks = batch["patient_week"]

                # Forward pass
                fvc, sigma = model(
                    img_axial,
                    img_coronal,
                    tabular,
                    week=week,
                    baseline_fvc=baseline_fvc,
                )

                fvc_np = fvc.cpu().numpy()
                sigma_np = sigma.cpu().numpy()

                for i in range(len(patient_weeks)):
                    # Clip confidence to 70 as per metric definition for safety
                    conf_val = max(sigma_np[i], 70)

                    submission_rows.append(
                        {
                            "Patient_Week": patient_weeks[i],
                            "FVC": fvc_np[i],
                            "Confidence": conf_val,
                        }
                    )

        # Create DataFrame and Save
        submission_df = pd.DataFrame(submission_rows)

        # Ensure directory exists
        os.makedirs("submission", exist_ok=True)
        save_path = os.path.join("submission", "submission.csv")

        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
        print(f"Total predictions: {len(submission_df)}")
        print(submission_df.head())

    else:
        print(
            f"\n=== Skipping Submission (Metric {final_metric:.4f} <= {THRESHOLD:.4f}) ==="
        )


if __name__ == "__main__":
    main()
