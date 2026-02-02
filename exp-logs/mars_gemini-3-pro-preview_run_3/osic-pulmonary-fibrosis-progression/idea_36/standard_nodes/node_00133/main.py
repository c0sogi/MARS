import os
import sys
import pandas as pd
import numpy as np
import torch
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import get_dataloaders, process_patient_images, AGE_MEAN, AGE_STD
from library.model import DSPRNet
from library.train import run_training


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Training
    # run_training handles the loop and saves the best model to checkpoints/best_model.pth
    print("--- Starting Training ---")
    run_training()

    # 3. Validation & Failure Analysis
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load Best Model
    model = DSPRNet().to(device)
    checkpoint_path = os.path.join(Config.WORKING_DIR, "checkpoints", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Get Validation Loader
    _, val_loader = get_dataloaders(Config.TRAIN_CSV, Config.VAL_CSV)

    # Inference Loop
    all_mu_raw = []
    all_sigma_raw = []
    all_targets_raw = []
    all_tabular_np = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets_raw = batch["target_raw"].to(device)

            mu, sigma = model(imgs, tabular)

            # Inverse Transform Predictions
            mu_raw = mu * Config.TARGET_STD + Config.TARGET_MEAN
            sigma_raw = sigma * Config.TARGET_STD

            all_mu_raw.append(mu_raw.cpu().numpy())
            all_sigma_raw.append(sigma_raw.cpu().numpy())
            all_targets_raw.append(targets_raw.cpu().numpy())
            all_tabular_np.append(tabular.cpu().numpy())

    # Concatenate results
    y_pred = np.concatenate(all_mu_raw)
    sigma_pred = np.concatenate(all_sigma_raw)
    y_true = np.concatenate(all_targets_raw)
    tabular_data = np.concatenate(all_tabular_np)

    # Calculate Metric
    metric = laplace_log_likelihood_metric(y_true, y_pred, sigma_pred)
    # Print full precision as requested
    print(f"Final Validation Metric: {metric}")

    # Failure Analysis
    # Calculate Absolute Error
    abs_error = np.abs(y_true - y_pred)

    # Tabular features index mapping from library.data.OSICDataset:
    # 0: Base FVC (Scaled), 1: Rel Time, 2: Age (Scaled), 3: Sex, 4: Smoking

    print("\nFailure Analysis (Correlation with Absolute Error):")
    feat_names = ["Baseline FVC", "Relative Time", "Age"]
    feat_indices = [0, 1, 2]

    for name, idx in zip(feat_names, feat_indices):
        # Compute Pearson correlation
        feat_vals = tabular_data[:, idx]
        corr = np.corrcoef(abs_error, feat_vals)[0, 1]
        print(f"  {name}: {corr:.6f}")

    # 4. Submission
    THRESHOLD = -6.573619738753321
    if metric > THRESHOLD:
        print("\n--- Metric threshold met. Generating submission... ---")
        generate_submission(model, device)
    else:
        print(
            f"\nMetric {metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


def generate_submission(model, device):
    # Load Metadata
    test_df = pd.read_csv(Config.TEST_CSV)
    sample_sub = pd.read_csv(Config.SUBMISSION_SAMPLE_CSV)

    # Prepare Output Directory
    sub_dir = os.path.join("submission")
    os.makedirs(sub_dir, exist_ok=True)

    # Preprocess Test Images (Cache once per patient)
    print("Processing test images...")
    test_patients = test_df["Patient"].unique()
    patient_img_tensors = {}

    # Cache directory for test images
    test_cache_dir = os.path.join(Config.WORKING_DIR, "test_cache")

    for patient in test_patients:
        # Get image path from metadata
        row = test_df[test_df["Patient"] == patient].iloc[0]
        img_path_rel = row["image_path"]

        # Process and load
        img_vol = process_patient_images(
            patient, img_path_rel, test_cache_dir, load_cached_data=True
        )

        # Convert to Tensor (C, H, W) and add batch dim
        img_tensor = (
            torch.tensor(np.transpose(img_vol, (2, 0, 1)), dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )
        patient_img_tensors[patient] = img_tensor

    # Prepare Baseline Lookup
    baseline_lookup = {}
    for _, row in test_df.iterrows():
        baseline_lookup[row["Patient"]] = row

    # Parse Sample Submission
    # We need to predict for every row in sample_submission
    sample_sub["Patient_ID"] = sample_sub["Patient_Week"].apply(
        lambda x: x.split("_")[0]
    )
    sample_sub["Week_Num"] = sample_sub["Patient_Week"].apply(
        lambda x: int(x.split("_")[1])
    )

    results = []

    model.eval()

    # Group by patient to batch predictions
    print("Running inference on test set...")
    with torch.no_grad():
        for patient, group in sample_sub.groupby("Patient_ID"):
            if patient not in baseline_lookup:
                continue

            base_info = baseline_lookup[patient]
            img_tensor = patient_img_tensors[patient]  # (1, 3, H, W)

            # Prepare Tabular Features for the whole group
            weeks = group["Week_Num"].values
            batch_size = len(weeks)

            # 1. Baseline FVC (Scaled)
            base_fvc_scaled = (
                base_info["FVC"] - Config.TARGET_MEAN
            ) / Config.TARGET_STD

            # 2. Relative Time (Vectorized)
            base_week = base_info["Weeks"]
            rel_times = (weeks - base_week) * Config.TIME_SCALE

            # 3. Age (Scaled)
            age_scaled = (base_info["Age"] - AGE_MEAN) / AGE_STD

            # 4. Sex
            sex_val = 0.0 if base_info["Sex"] == "Male" else 1.0

            # 5. Smoking
            smoke_map = {"Never smoked": 0.0, "Ex-smoker": 1.0, "Currently smokes": 2.0}
            smoke_val = smoke_map.get(base_info["SmokingStatus"], 0.0)

            # Construct Tabular Tensor
            # Shape: (Batch, 5)
            tabular_np = np.zeros((batch_size, 5), dtype=np.float32)
            tabular_np[:, 0] = base_fvc_scaled
            tabular_np[:, 1] = rel_times
            tabular_np[:, 2] = age_scaled
            tabular_np[:, 3] = sex_val
            tabular_np[:, 4] = smoke_val

            tabular_tensor = torch.tensor(tabular_np, dtype=torch.float32).to(device)

            # Expand Image Tensor
            img_batch = img_tensor.repeat(batch_size, 1, 1, 1)

            # Inference
            mu, sigma = model(img_batch, tabular_tensor)

            # Inverse Transform
            mu_raw = mu.cpu().numpy() * Config.TARGET_STD + Config.TARGET_MEAN
            sigma_raw = sigma.cpu().numpy() * Config.TARGET_STD

            # Apply Confidence Clipping (Max(sigma, 70))
            sigma_clipped = np.maximum(sigma_raw, 70.0)

            # Store Results
            patient_weeks = group["Patient_Week"].values
            for i in range(batch_size):
                results.append(
                    {
                        "Patient_Week": patient_weeks[i],
                        "FVC": mu_raw[i],
                        "Confidence": sigma_clipped[i],
                    }
                )

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure strict ordering matching sample_submission
    final_submission = pd.merge(
        sample_sub[["Patient_Week"]], submission_df, on="Patient_Week", how="left"
    )

    # Save
    out_path = os.path.join(sub_dir, "submission.csv")
    final_submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")


if __name__ == "__main__":
    main()
