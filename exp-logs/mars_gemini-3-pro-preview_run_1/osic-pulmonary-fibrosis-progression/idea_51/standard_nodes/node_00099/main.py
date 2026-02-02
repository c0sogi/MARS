import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.dataset import LungDataset
from library.model import NSLHN
from library.engine import run_training


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Adjust configuration for a fast baseline run
    # We reduce epochs to ensure runtime is well within limits while allowing convergence
    Config.EPOCHS = 20

    print(f"Initializing run with {Config.EPOCHS} epochs...")

    # 2. Training
    # run_training handles the loop, early stopping, and saving best_model.pth
    best_val_score = run_training(debug=False)

    # 3. Validation & Metric Calculation
    print("\nRunning Final Validation...")

    # Load the best model
    model = NSLHN().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found. Training may have failed.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Prepare Validation Loader
    val_df = pd.read_csv(Config.VAL_CSV)
    val_dataset = LungDataset(val_df, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Containers for Failure Analysis
    all_targets = []
    all_preds = []
    all_sigmas = []
    all_errors = []

    # Features for correlation: Age, Sex, Smoking, Percent, TimeDelta
    # Tabular in dataset is: [age_norm, sex_enc, smoke_enc, percent_norm]
    feature_data = {"Age": [], "Sex": [], "Smoking": [], "Percent": [], "TimeDelta": []}

    with torch.no_grad():
        for batch in val_loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            time_delta = batch["time_delta"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            target = batch["target"].to(device)

            # Forward Pass
            outputs = model(img_axial, img_coronal, tabular)

            # Unpack Parameters
            alpha = outputs[:, 0:1]
            sigma_base = outputs[:, 1:2]
            sigma_growth = outputs[:, 2:3]

            # Reconstruct Trajectory
            fvc_pred = baseline_fvc + alpha * time_delta
            sigma_pred = sigma_base + sigma_growth * torch.abs(time_delta)

            # Store for metric
            all_targets.extend(target.cpu().numpy().flatten())
            all_preds.extend(fvc_pred.cpu().numpy().flatten())
            all_sigmas.extend(sigma_pred.cpu().numpy().flatten())

            # Store for failure analysis
            batch_errors = torch.abs(target - fvc_pred).cpu().numpy().flatten()
            all_errors.extend(batch_errors)

            # Extract features (undo normalization for interpretability where possible, or keep raw)
            # tabular: [age_norm, sex, smoke, percent_norm]
            tab_np = tabular.cpu().numpy()
            feature_data["Age"].extend(tab_np[:, 0])
            feature_data["Sex"].extend(tab_np[:, 1])
            feature_data["Smoking"].extend(tab_np[:, 2])
            feature_data["Percent"].extend(tab_np[:, 3])
            feature_data["TimeDelta"].extend(time_delta.cpu().numpy().flatten())

    # Calculate Final Metric
    final_metric = calculate_metric(all_targets, all_preds, all_sigmas)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nFailure Analysis (Correlation with Absolute Error):")
    errors = np.array(all_errors)

    for feat_name, feat_vals in feature_data.items():
        vals = np.array(feat_vals)
        # Handle constant arrays to avoid warnings
        if np.std(vals) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(vals, errors)
        print(f"  {feat_name}: {corr:.4f}")

    # 5. Submission Generation
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print("\nMetric passed threshold. Generating submission...")

        test_df = pd.read_csv(Config.TEST_CSV)
        test_dataset = LungDataset(test_df, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        submission_rows = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                img_axial = batch["img_axial"].to(device)
                img_coronal = batch["img_coronal"].to(device)
                tabular = batch["tabular"].to(device)
                time_delta = batch["time_delta"].to(device)
                baseline_fvc = batch["baseline_fvc"].to(device)
                # patient_id is a list of strings
                patient_ids = batch["patient_id"]

                # Forward Pass
                outputs = model(img_axial, img_coronal, tabular)

                alpha = outputs[:, 0:1]
                sigma_base = outputs[:, 1:2]
                sigma_growth = outputs[:, 2:3]

                # Calculate Predictions
                fvc_pred = baseline_fvc + alpha * time_delta
                sigma_pred = sigma_base + sigma_growth * torch.abs(time_delta)

                # Clip Confidence
                sigma_pred = torch.clamp(sigma_pred, min=70)

                # Move to CPU
                fvc_np = fvc_pred.cpu().numpy().flatten()
                sigma_np = sigma_pred.cpu().numpy().flatten()

                # We need to reconstruct the Patient_Week ID
                # The test dataset provides time_delta which is (Predict_Week - Baseline_Week)
                # But we need the actual Predict_Week to form the ID.
                # In dataset.py for test mode: meta_dt = Predict_Week - Baseline_Week
                # We can retrieve Predict_Week from the dataframe using the batch index,
                # but simpler is to rely on the fact that test.csv in metadata has Patient_Week

                # Get the corresponding rows from the original dataframe
                start_idx = i * Config.BATCH_SIZE
                end_idx = start_idx + len(patient_ids)
                batch_df_rows = test_df.iloc[start_idx:end_idx]

                for idx, row_id in enumerate(batch_df_rows["Patient_Week"]):
                    submission_rows.append(
                        {
                            "Patient_Week": row_id,
                            "FVC": fvc_np[idx],
                            "Confidence": sigma_np[idx],
                        }
                    )

        # Save Submission
        sub_df = pd.DataFrame(submission_rows)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH} with {len(sub_df)} rows.")

    else:
        print(
            f"\nMetric {final_metric} did not pass threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
