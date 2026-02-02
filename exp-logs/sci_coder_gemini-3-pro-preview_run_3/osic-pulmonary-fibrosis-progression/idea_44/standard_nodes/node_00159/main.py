import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

# Import library modules
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import get_dataloaders, CTPreprocessor, ClinicalPreprocessor
from library.model import DSPRNet
from library.train import Runner


def main():
    # 1. Setup and Configuration Overrides
    # Limit epochs for fast baseline execution
    Config.EPOCHS = 25
    seed_everything(Config.SEED)

    print("Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")

    # 2. Training
    print("\n--- Starting Training ---")
    runner = Runner()
    runner.train()

    # 3. Validation and Failure Analysis
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load best model
    model = DSPRNet().to(Config.DEVICE)
    model.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    # Get validation loader
    _, val_loader = get_dataloaders()

    # Containers for analysis
    all_mu = []
    all_sigma = []
    all_targets = []
    all_clinical_raw = []  # To store features for correlation analysis

    # We need to access the original dataframe to get raw feature values for correlation
    # The val_loader dataset has the preprocessed dataframe.
    # We can reconstruct necessary info or just use the preprocessed values which are linear transformations.
    # Let's collect the inputs passed to the model.

    with torch.no_grad():
        for images, clinical, targets in val_loader:
            images = images.to(Config.DEVICE)
            clinical = clinical.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            # Forward pass
            (mu, sigma), _ = model(images, clinical)

            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_clinical_raw.append(clinical.cpu().numpy())

    # Concatenate
    mu_scaled = np.concatenate(all_mu)
    sigma_scaled = np.concatenate(all_sigma)
    targets_scaled = np.concatenate(all_targets)
    clinical_inputs = np.concatenate(all_clinical_raw)

    # Inverse Transform
    # Target/Preds: value * std + mean
    mu_orig = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN
    sigma_orig = sigma_scaled * Config.TARGET_STD
    targets_orig = targets_scaled * Config.TARGET_STD + Config.TARGET_MEAN

    # Calculate Metric
    final_metric = laplace_log_likelihood_metric(targets_orig, mu_orig, sigma_orig)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate absolute error
    abs_error = np.abs(targets_orig - mu_orig)

    # Clinical inputs shape: [Baseline_FVC_Scaled, Relative_Time, Age_Scaled, Sex_Code, Smoking_Code]
    # We can correlate error with these.
    analysis_df = pd.DataFrame(
        {
            "Abs_Error": abs_error,
            "Baseline_FVC_Scaled": clinical_inputs[:, 0],
            "Relative_Time": clinical_inputs[:, 1],
            "Age_Scaled": clinical_inputs[:, 2],
            "Sex_Code": clinical_inputs[:, 3],
            "Smoking_Code": clinical_inputs[:, 4],
            "Target_FVC": targets_orig,
        }
    )

    print("\nFailure Analysis (Correlation with Absolute Error):")
    correlations = analysis_df.corr()["Abs_Error"].sort_values(ascending=False)
    print(correlations)

    # 4. Submission
    THRESHOLD = -6.573619738753321
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


def generate_submission(model):
    """
    Generates submission.csv for the test set.
    """
    model.eval()

    # Load Metadata
    test_meta = pd.read_csv(Config.TEST_CSV)
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # Prepare Image Processor
    img_processor = CTPreprocessor()

    # Parse Sample Submission to get Patient and Week
    # Format: ID..._WeekNum
    sample_sub["Patient"] = sample_sub["Patient_Week"].apply(lambda x: x.split("_")[0])
    sample_sub["Weeks"] = sample_sub["Patient_Week"].apply(
        lambda x: int(x.split("_")[1])
    )

    # We process patient by patient to avoid reloading images
    unique_patients = sample_sub["Patient"].unique()

    predictions = []

    print(f"Processing {len(unique_patients)} patients for submission...")

    with torch.no_grad():
        for patient_id in unique_patients:
            # Get patient baseline info
            patient_info = test_meta[test_meta["Patient"] == patient_id].iloc[0]

            # Load Image
            image_dir = patient_info["image_path"]
            image = img_processor.process_patient(
                patient_id, image_dir, load_cached_data=True
            )
            image_tensor = (
                torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
            )
            # image_tensor shape: (1, 3, 260, 260)

            # Get all requested weeks for this patient
            patient_reqs = sample_sub[sample_sub["Patient"] == patient_id].copy()
            requested_weeks = patient_reqs["Weeks"].values

            # Prepare Clinical Features Batch
            # Need: [Baseline_FVC_Scaled, Relative_Time, Age_Scaled, Sex_Code, Smoking_Code]

            # 1. Baseline FVC Scaled
            base_fvc = patient_info["FVC"]  # In test.csv, FVC is the baseline FVC
            base_fvc_scaled = (base_fvc - Config.TARGET_MEAN) / Config.TARGET_STD

            # 2. Age Scaled
            age_scaled = (patient_info["Age"] - Config.AGE_MEAN) / Config.AGE_STD

            # 3. Sex Code
            sex_code = 1 if patient_info["Sex"] == "Female" else 0

            # 4. Smoking Code
            smoke_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
            smoke_code = smoke_map.get(patient_info["SmokingStatus"], 0)

            # 5. Relative Time (Vectorized)
            # Relative_Time = (Current_Week - Baseline_Week) * Scale
            # In test.csv, 'Weeks' is the baseline week
            base_week = patient_info["Weeks"]
            rel_times = (requested_weeks - base_week) * Config.TIME_SCALE

            # Construct Batch
            n_samples = len(requested_weeks)

            # Expand static features
            batch_base_fvc = np.full(n_samples, base_fvc_scaled)
            batch_age = np.full(n_samples, age_scaled)
            batch_sex = np.full(n_samples, sex_code)
            batch_smoke = np.full(n_samples, smoke_code)

            clinical_np = np.stack(
                [batch_base_fvc, rel_times, batch_age, batch_sex, batch_smoke], axis=1
            ).astype(np.float32)

            clinical_tensor = torch.tensor(clinical_np).to(Config.DEVICE)

            # Expand Image Tensor
            image_batch = image_tensor.expand(n_samples, -1, -1, -1)

            # Inference
            # Process in chunks if too large (though 150 weeks is fine for batch inference)
            (mu, sigma), _ = model(image_batch, clinical_tensor)

            # Inverse Transform
            mu = mu.cpu().numpy()
            sigma = sigma.cpu().numpy()

            mu_orig = mu * Config.TARGET_STD + Config.TARGET_MEAN
            sigma_orig = sigma * Config.TARGET_STD  # Scale only

            # Post-processing
            sigma_final = np.maximum(sigma_orig, 70)

            # Store
            for i, week in enumerate(requested_weeks):
                pw_id = f"{patient_id}_{week}"
                predictions.append(
                    {
                        "Patient_Week": pw_id,
                        "FVC": mu_orig[i],
                        "Confidence": sigma_final[i],
                    }
                )

    # Create DataFrame
    sub_df = pd.DataFrame(predictions)

    # Ensure order matches sample_submission (optional but good practice)
    # We can just merge or reindex.
    final_sub = sample_sub[["Patient_Week"]].merge(
        sub_df, on="Patient_Week", how="left"
    )

    # Fill NaNs if any (shouldn't be)
    final_sub["FVC"] = final_sub["FVC"].fillna(2000)
    final_sub["Confidence"] = final_sub["Confidence"].fillna(100)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    final_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
