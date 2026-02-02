import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders, process_patient_images, SEX_MAP, SMOKING_MAP
from library.model import DSPRNet
from library.train import run_training

# Override Config for fast baseline execution
Config.EPOCHS = 30
Config.DEBUG = False


def generate_submission(model, stats, device):
    """
    Generates submission file for the test set.
    Predicts FVC and Confidence for every Patient_Week in sample_submission.csv.
    """
    print("Generating submission...")

    # Load necessary files
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Parse Patient and Weeks from sample submission
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

    unique_patients = sub_df["Patient"].unique()

    # Extract normalization stats
    fvc_mean = stats["fvc_mean"]
    fvc_std = stats["fvc_std"]
    age_mean = stats["age_mean"]
    age_std = stats["age_std"]

    results = []
    model.eval()

    with torch.no_grad():
        for patient in unique_patients:
            # Get baseline metadata for this patient
            if patient not in test_df["Patient"].values:
                continue

            base_row = test_df[test_df["Patient"] == patient].iloc[0]

            # 1. Prepare Image Input
            # Load and process the image volume (Anchor + 2 boundaries)
            img_vol = process_patient_images(
                patient, base_row["image_path"], Config.CACHE_DIR
            )
            # Shape: (3, H, W) -> Add batch dim -> (1, 3, H, W)
            img_tensor = (
                torch.tensor(img_vol, dtype=torch.float32).unsqueeze(0).to(device)
            )

            # 2. Prepare Tabular Inputs
            base_fvc = base_row["FVC"]
            base_week = base_row["Weeks"]
            base_age = base_row["Age"]
            sex = base_row["Sex"]
            smoking = base_row["SmokingStatus"]

            # Encode Categoricals
            sex_enc = SEX_MAP.get(sex, 0)
            smoke_enc = SMOKING_MAP.get(smoking, 1)
            smoke_oh = [0, 0, 0]
            smoke_oh[smoke_enc] = 1

            # Normalize Continuous Features
            base_fvc_norm = (base_fvc - fvc_mean) / fvc_std
            age_norm = (base_age - age_mean) / age_std

            # Identify all target weeks for this patient
            patient_weeks = sub_df[sub_df["Patient"] == patient]["Weeks"].values

            # Construct batch of tabular features
            # Feature Vector: [Base_FVC, t_rel, Age, Sex, Smoke_0, Smoke_1, Smoke_2]
            batch_tabular = []
            for w in patient_weeks:
                # Calculate relative time
                t_rel = (w - base_week) * Config.TIME_SCALER
                feats = [base_fvc_norm, t_rel, age_norm, sex_enc] + smoke_oh
                batch_tabular.append(feats)

            # Convert to tensor
            batch_tabular = torch.tensor(
                np.array(batch_tabular), dtype=torch.float32
            ).to(device)

            # Expand image tensor to match the number of weeks (batch size)
            batch_size = len(patient_weeks)
            batch_imgs = img_tensor.expand(batch_size, -1, -1, -1)

            # 3. Predict
            preds = model(batch_imgs, batch_tabular)

            # 4. Inverse Transform
            mu_norm = preds[:, 0]
            raw_sigma = preds[:, 1]

            # Sigma: Softplus -> Scale
            sigma_norm = F.softplus(raw_sigma)
            sigma_real = (sigma_norm * fvc_std).cpu().numpy()

            # Mu: Un-normalize
            mu_real = (mu_norm * fvc_std + fvc_mean).cpu().numpy()

            # Clip Sigma according to metric rules for submission
            sigma_final = np.maximum(sigma_real, Config.MIN_CONFIDENCE)

            # 5. Store Results
            for i, w in enumerate(patient_weeks):
                pw_id = f"{patient}_{w}"
                results.append(
                    {
                        "Patient_Week": pw_id,
                        "FVC": mu_real[i],
                        "Confidence": sigma_final[i],
                    }
                )

    # Save Submission
    submission = pd.DataFrame(results)
    # Ensure columns are in correct order
    submission = submission[["Patient_Week", "FVC", "Confidence"]]
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup_directories()

    # 2. Train Model
    # run_training handles the training loop and saves the best model to Config.BEST_MODEL_PATH
    print("Starting training pipeline...")
    run_training()

    # 3. Validation & Analysis
    print("\nStarting validation and failure analysis...")

    # Load data loaders to get the exact stats used during training
    _, val_loader, _, stats = get_dataloaders()

    # Load the best model
    device = Config.DEVICE
    model = DSPRNet().to(device)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Statistics for un-normalization
    fvc_mean = stats["fvc_mean"]
    fvc_std = stats["fvc_std"]

    val_metrics_data = []
    val_analysis_data = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)

            # Predict
            preds = model(images, tabular)

            # Un-normalize
            mu_norm = preds[:, 0]
            raw_sigma = preds[:, 1]

            sigma_norm = F.softplus(raw_sigma)
            sigma_real = sigma_norm * fvc_std
            mu_real = mu_norm * fvc_std + fvc_mean
            target_real = targets * fvc_std + fvc_mean

            # Convert to numpy
            mu_np = mu_real.cpu().numpy()
            sigma_np = sigma_real.cpu().numpy()
            target_np = target_real.cpu().numpy()
            tab_np = tabular.cpu().numpy()

            # Store for global metric calculation
            val_metrics_data.append((target_np, mu_np, sigma_np))

            # Store for failure analysis
            for i in range(len(target_np)):
                error = np.abs(target_np[i] - mu_np[i])
                # Tabular: [Base_FVC, t_rel, Age, Sex, Smoke_0, Smoke_1, Smoke_2]
                val_analysis_data.append(
                    {
                        "Error": error,
                        "Base_FVC_Norm": tab_np[i, 0],
                        "Time_Rel": tab_np[i, 1],
                        "Age_Norm": tab_np[i, 2],
                        "Target": target_np[i],
                        "Sigma": sigma_np[i],
                    }
                )

    # Calculate Final Metric
    all_targets = np.concatenate([x[0] for x in val_metrics_data])
    all_mus = np.concatenate([x[1] for x in val_metrics_data])
    all_sigmas = np.concatenate([x[2] for x in val_metrics_data])

    final_metric = calculate_metric(all_targets, all_mus, all_sigmas)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    df_analysis = pd.DataFrame(val_analysis_data)
    print("\nFailure Analysis (Correlation with Error):")
    correlations = df_analysis.corr()["Error"].sort_values(ascending=False)
    print(correlations)

    # 4. Submission
    # Threshold defined in task
    THRESHOLD = -6.573619738753321

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Proceeding to submission."
        )
        generate_submission(model, stats, device)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
