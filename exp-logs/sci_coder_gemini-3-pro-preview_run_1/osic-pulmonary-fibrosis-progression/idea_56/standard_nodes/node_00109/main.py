import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything, score
from library.data import LungDataset, get_transforms
from library.model import BBSLNet
from library.train import run_training


def main():
    # ==========================================
    # 1. Train Model
    # ==========================================
    print("Starting Training Pipeline...")
    # run_training handles the full training loop and saves the best model to Config.CHECKPOINT_PATH
    # We use debug=False for the final run to ensure good performance.
    best_val_score = run_training(debug=False)

    # ==========================================
    # 2. Validation Inference & Metric Calculation
    # ==========================================
    print("\nRunning Validation Inference...")
    device = torch.device(Config.DEVICE)

    # Load Best Model
    model = BBSLNet().to(device)
    checkpoint_path = Config.CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Setup Validation Loader
    val_dataset = LungDataset(
        csv_path=Config.VAL_CSV, mode="val", transform=get_transforms("val")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Collect Predictions
    all_true_fvc = []
    all_pred_fvc = []
    all_pred_sigma = []

    # For Failure Analysis
    val_df = pd.read_csv(Config.VAL_CSV)

    with torch.no_grad():
        for batch in val_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            meta = batch["meta"].to(device)

            baseline_fvcs = batch["baseline_fvc"].to(device)
            delta_weeks = batch["delta_week"].to(device)

            # Forward Pass
            outputs = model(img_ax, img_cor, meta)

            alpha = outputs[:, 0]
            sigma_base = outputs[:, 1]
            sigma_growth = outputs[:, 2]

            # Compute Predictions
            fvc_pred = baseline_fvcs + alpha * delta_weeks
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_weeks)

            # Store
            all_true_fvc.extend(batch["fvc"].numpy())
            all_pred_fvc.extend(fvc_pred.cpu().numpy())
            all_pred_sigma.extend(sigma_pred.cpu().numpy())

    # Calculate Final Metric
    final_metric = score(all_true_fvc, all_pred_fvc, all_pred_sigma)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 3. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")
    val_df["pred_fvc"] = all_pred_fvc
    val_df["abs_error"] = np.abs(val_df["FVC"] - val_df["pred_fvc"])

    # Map categorical features for correlation
    sex_map = {"Male": 0, "Female": 1}
    smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    val_df["Sex_Num"] = val_df["Sex"].map(sex_map)
    val_df["Smoke_Num"] = val_df["SmokingStatus"].map(smoke_map)

    # Calculate correlations
    analysis_cols = ["Weeks", "Percent", "Age", "Sex_Num", "Smoke_Num"]
    print("Correlation between Absolute Error and Input Features:")
    for col in analysis_cols:
        if col in val_df.columns:
            corr = val_df[col].corr(val_df["abs_error"])
            print(f"{col}: {corr:.4f}")

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        generate_submission(model, device)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission Skipped."
        )


def generate_submission(model, device):
    """
    Generates submission.csv for the test set.
    Optimized to run inference once per patient.
    """
    test_meta_path = Config.TEST_CSV
    df_test = pd.read_csv(test_meta_path)

    # 1. Prepare Unique Patient Data for Inference
    # We only need one prediction of parameters (alpha, sigma_base, sigma_growth) per patient
    unique_patients = df_test.drop_duplicates(subset=["Patient"]).copy()

    # Rename columns to match LungDataset expectations
    # LungDataset expects: Percent, Age, Sex, SmokingStatus
    # metadata/test.csv has: Baseline_Percent, Baseline_Age, etc.
    rename_map = {
        "Baseline_Percent": "Percent",
        "Baseline_Age": "Age",
        "Baseline_Sex": "Sex",
        "Baseline_SmokingStatus": "SmokingStatus",
    }
    unique_patients_renamed = unique_patients.rename(columns=rename_map)

    # Ensure 'Weeks' exists (LungDataset uses it for delta calculation, though model doesn't use it)
    if "Weeks" not in unique_patients_renamed.columns:
        unique_patients_renamed["Weeks"] = unique_patients_renamed["Baseline_Week"]

    # Save temp CSV for LungDataset
    temp_csv_path = os.path.join(Config.WORKING_DIR, "temp_test_unique.csv")
    unique_patients_renamed.to_csv(temp_csv_path, index=False)

    # 2. Run Inference
    test_ds = LungDataset(
        csv_path=temp_csv_path,
        mode="test",  # Mode doesn't strictly matter for LungDataset logic provided, but good for clarity
        transform=get_transforms("val"),  # No augmentation
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    patient_params = {}

    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            meta = batch["meta"].to(device)
            p_ids = batch["patient_id"]

            # Forward pass to get parameters
            outputs = model(img_ax, img_cor, meta)
            # outputs: [alpha, sigma_base, sigma_growth]

            outputs_np = outputs.cpu().numpy()

            for pid, params in zip(p_ids, outputs_np):
                patient_params[pid] = params

    # 3. Compute Final Predictions for All Rows
    submission_rows = []

    # Iterate over original test dataframe
    for idx, row in df_test.iterrows():
        pid = row["Patient"]

        # Default values
        fvc_pred = 2000
        conf_pred = 100

        if pid in patient_params:
            alpha, sigma_base, sigma_growth = patient_params[pid]

            # Calculate delta week for this specific prediction
            delta_week = row["Predict_Week"] - row["Baseline_Week"]

            # Apply Formula
            fvc_pred = row["Baseline_FVC"] + alpha * delta_week
            conf_pred = sigma_base + sigma_growth * abs(delta_week)

            # Clip Confidence
            conf_pred = max(conf_pred, 70.0)

        submission_rows.append(
            {
                "Patient_Week": row["Patient_Week"],
                "FVC": fvc_pred,
                "Confidence": conf_pred,
            }
        )

    # 4. Save Submission
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Cleanup
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)


if __name__ == "__main__":
    main()
