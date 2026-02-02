import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.model import MACAN
from library.data import get_dataloaders


def predict_test():
    """
    Generates predictions for the test set and saves submission.csv.

    Steps:
    1. Load trained MACAN model.
    2. Load test data using the existing data loader.
    3. Load sample_submission.csv to identify target weeks.
    4. For each patient:
       - Expand baseline data to cover all requested weeks.
       - Update relative time features.
       - Run inference.
       - Inverse transform predictions.
    5. Save formatted submission file.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running inference on device: {device}")

    # 2. Load Model
    model = MACAN().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading model weights from {Config.BEST_MODEL_PATH}")
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Warning: Best model weights not found. Using random initialization.")

    model.eval()

    # 3. Load Data
    # We use batch_size=1 to process one patient at a time, facilitating
    # the expansion of that patient's data to multiple weeks.
    _, _, test_loader = get_dataloaders(batch_size=1, num_workers=Config.NUM_WORKERS)

    # Load sample submission to know which weeks to predict
    if not os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Sample submission not found at {Config.SAMPLE_SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # List to store results dictionaries
    results = []

    print("Starting inference loop...")

    with torch.no_grad():
        for batch in test_loader:
            # Extract baseline data for the patient
            # Batch size is 1, so we take the first element
            patient_id = batch["patient_id"][0]
            image = batch["image"].to(device)  # Shape: (1, 3, H, W)
            base_tabular = batch["tabular"].to(device)  # Shape: (1, 8)
            base_week = batch["weeks"].item()  # Scalar

            # Identify all rows in submission file for this patient
            # Format of Patient_Week is "ID..._WeekNum"
            mask = sub_df["Patient_Week"].str.startswith(patient_id + "_")
            patient_sub_rows = sub_df[mask]

            if len(patient_sub_rows) == 0:
                continue

            # Extract the target weeks from the "Patient_Week" string
            # Example: "ID123_10" -> 10
            target_weeks = (
                patient_sub_rows["Patient_Week"]
                .apply(lambda x: int(x.split("_")[-1]))
                .values
            )

            if len(target_weeks) == 0:
                continue

            # --- Prepare Batch for Inference ---
            n_samples = len(target_weeks)

            # 1. Replicate Image: (N, 3, H, W)
            batch_images = image.repeat(n_samples, 1, 1, 1)

            # 2. Replicate Tabular Features: (N, 8)
            batch_tabular = base_tabular.repeat(n_samples, 1)

            # 3. Update Relative Time Feature
            # Tabular Index 1 is "Scaled_Rel_Weeks"
            # Formula: (Target_Week - Base_Week) * TIME_SCALE
            rel_weeks_raw = target_weeks - base_week
            rel_weeks_scaled = rel_weeks_raw * Config.TIME_SCALE

            # Update the tensor (ensure correct device and dtype)
            batch_tabular[:, 1] = torch.tensor(
                rel_weeks_scaled, dtype=torch.float32, device=device
            )

            # --- Run Inference ---
            mu_scaled, sigma_scaled = model(batch_images, batch_tabular)

            # --- Inverse Transformation ---
            # Convert tensors to numpy
            mu_scaled = mu_scaled.cpu().numpy()
            sigma_scaled = sigma_scaled.cpu().numpy()

            # FVC = mu_scaled * STD + MEAN
            mu_real = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN

            # Confidence = sigma_scaled * STD
            sigma_real = sigma_scaled * Config.TARGET_STD

            # --- Collect Results ---
            for i, week in enumerate(target_weeks):
                pid_week = f"{patient_id}_{week}"
                fvc_pred = mu_real[i]
                conf_pred = sigma_real[i]

                # Apply Confidence Clip (min 70 ml)
                conf_pred = max(conf_pred, Config.CONFIDENCE_CLIP)

                results.append(
                    {"Patient_Week": pid_week, "FVC": fvc_pred, "Confidence": conf_pred}
                )

    # 4. Finalize Submission
    pred_df = pd.DataFrame(results)

    # Merge predictions into the sample submission template
    # This ensures the order matches exactly and handles any missing patients gracefully
    final_sub = sub_df[["Patient_Week"]].merge(pred_df, on="Patient_Week", how="left")

    # Fill NaN values with defaults (safe fallback)
    # 2000 ml is a generic FVC, 100 is a generic confidence
    final_sub["FVC"] = final_sub["FVC"].fillna(2000)
    final_sub["Confidence"] = final_sub["Confidence"].fillna(100)

    # Save to disk
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    final_sub.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total rows predicted: {len(final_sub)}")
