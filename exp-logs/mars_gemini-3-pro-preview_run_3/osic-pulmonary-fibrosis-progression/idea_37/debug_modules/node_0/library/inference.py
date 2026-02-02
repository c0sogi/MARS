import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.data import get_test_loader
from library.model import RCOSRNet


def predict_test_set(model=None, device=None):
    """
    Generates submission.csv by predicting FVC for all Patient_Week combinations
    in sample_submission.csv.

    Args:
        model (nn.Module, optional): Trained RCOSRNet model. If None, loads from Config.CHECKPOINT_DIR.
        device (torch.device, optional): Torch device. If None, uses Config.DEVICE.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    # 1. Setup
    if device is None:
        device = Config.DEVICE

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Load model if not provided
    if model is None:
        model = RCOSRNet().to(device)
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading model from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(
                "Warning: No checkpoint found. Using initialized model (random weights)."
            )

    model.eval()
    print("Generating submission...")

    # 2. Load Submission Template
    sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
    if not os.path.exists(sample_sub_path):
        # Fallback for case sensitivity
        sample_sub_path = os.path.join(Config.INPUT_DIR, "sampleSubmission.csv")

    sub_df = pd.read_csv(sample_sub_path)

    # 3. Pre-load Test Patient Data
    # We use the test loader to get processed images and baseline clinical features.
    # Batch size 1 ensures we handle one patient at a time for dictionary construction.
    test_loader = get_test_loader(batch_size=1, num_workers=0)

    patient_data = {}
    with torch.no_grad():
        for batch in test_loader:
            pid = batch["patient_id"][0]
            # Store tensors on CPU initially to save GPU memory
            patient_data[pid] = {
                "image": batch["image"],  # (1, 3, H, W)
                "base_week": batch["base_week"].item(),
                "clinical_base": batch["clinical"],  # (1, 5) where time is ~0
            }

    # 4. Parse Submission File
    # Format: ID00419637202311204720264_6
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

    final_preds = []

    # 5. Inference Loop (Grouped by Patient)
    unique_patients = sub_df["Patient"].unique()

    for pid in unique_patients:
        if pid not in patient_data:
            # Skip patients not found in the test set metadata
            continue

        # Get all requested weeks for this patient
        group = sub_df[sub_df["Patient"] == pid]
        weeks = group["Weeks"].values
        n_samples = len(weeks)
        p_data = patient_data[pid]

        # A. Prepare Image Batch
        # Repeat the single patient image for all requested weeks
        # (N, 3, H, W)
        img_tensor = p_data["image"].to(device)
        imgs = img_tensor.repeat(n_samples, 1, 1, 1)

        # B. Prepare Clinical Batch
        # Base: [Base_FVC_Std, Time=0, Age_Std, Sex, Smoke]
        clinical_base = p_data["clinical_base"].to(device)
        clinicals = clinical_base.repeat(n_samples, 1)  # (N, 5)

        # Calculate relative time for each requested week
        base_week = p_data["base_week"]
        rel_times = weeks - base_week
        time_scaled = rel_times * Config.TIME_SCALE

        # Update Time column (index 1 is Relative Time)
        clinicals[:, 1] = torch.tensor(time_scaled, dtype=torch.float32, device=device)

        # C. Forward Pass
        with torch.no_grad():
            mu, sigma = model(imgs, clinicals)

        # D. Inverse Transformation
        mu = mu.cpu().numpy().flatten()
        sigma = sigma.cpu().numpy().flatten()

        # Un-Z-score
        mu_orig = mu * Config.TARGET_STD + Config.TARGET_MEAN
        sigma_orig = sigma * Config.TARGET_STD

        # E. Post-Processing
        # Clip confidence at 70ml as per metric definition
        sigma_orig = np.maximum(sigma_orig, Config.CONFIDENCE_CLIP)

        # F. Collect Results
        for i, week in enumerate(weeks):
            patient_week = f"{pid}_{week}"
            final_preds.append(
                {
                    "Patient_Week": patient_week,
                    "FVC": mu_orig[i],
                    "Confidence": sigma_orig[i],
                }
            )

    # 6. Save Submission
    submission = pd.DataFrame(final_preds)

    # Ensure correct column order
    submission = submission[["Patient_Week", "FVC", "Confidence"]]

    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

    return submission
