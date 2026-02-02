import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, TargetScaler
from library.data import get_test_dataloader
from library.model import GMARNet


def generate_submission():
    """
    Generates the submission file for the competition.

    Steps:
    1. Re-fit TargetScalers using training data to ensure correct inverse transformation.
    2. Load the trained GMAR-Net model.
    3. Process the test set using the sample submission template.
    4. Predict FVC (mu) and Confidence (sigma).
    5. Inverse transform predictions to original scale (ml).
    6. Apply post-processing constraints (Confidence clipping).
    7. Save to CSV.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("--- Starting Inference ---")

    # -------------------------------------------------------------------------
    # 1. Prepare Scalers
    # -------------------------------------------------------------------------
    # We must fit the scalers on the training data exactly as done during training
    # to correctly invert the Z-score standardization.
    print("Fitting scalers on training data...")
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Training metadata not found at {Config.TRAIN_CSV}")

    train_df = pd.read_csv(Config.TRAIN_CSV)

    fvc_scaler = TargetScaler()
    fvc_scaler.fit(train_df["FVC"].values)

    age_scaler = TargetScaler()
    age_scaler.fit(train_df["Age"].values)

    scalers = {"fvc_scaler": fvc_scaler, "age_scaler": age_scaler}

    # -------------------------------------------------------------------------
    # 2. Load Model
    # -------------------------------------------------------------------------
    print(f"Loading model from {Config.BEST_MODEL_PATH}...")
    model = GMARNet().to(device)

    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.BEST_MODEL_PATH}"
        )

    state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # -------------------------------------------------------------------------
    # 3. Prepare Data Loader
    # -------------------------------------------------------------------------
    print("Preparing test data loader...")
    # get_test_dataloader handles merging sample_submission with test metadata
    test_loader = get_test_dataloader(scalers)

    # -------------------------------------------------------------------------
    # 4. Inference Loop
    # -------------------------------------------------------------------------
    print("Running predictions...")

    patient_weeks = []
    fvc_preds = []
    confidence_preds = []

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            images = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            p_weeks = batch["patient_week"]

            # Forward pass
            # Output shape: (Batch, 2) -> [mu_scaled, sigma_scaled]
            preds = model(images, clinical)

            mu_scaled = preds[:, 0].cpu().numpy()
            sigma_scaled = preds[:, 1].cpu().numpy()

            # 5. Inverse Transformation
            # Inverse transform mu: mu * std + mean
            mu_original = fvc_scaler.inverse_transform(mu_scaled)

            # Inverse transform sigma: sigma * std (scale only, no mean shift)
            sigma_original = fvc_scaler.inverse_transform_sigma(sigma_scaled)

            # 6. Post-Processing
            # Clip sigma to minimum 70ml as per metric definition
            sigma_clipped = np.maximum(sigma_original, Config.METRIC_CLIP_SIGMA)

            # Collect results
            patient_weeks.extend(p_weeks)
            fvc_preds.extend(mu_original)
            confidence_preds.extend(sigma_clipped)

    # -------------------------------------------------------------------------
    # 7. Save Submission
    # -------------------------------------------------------------------------
    submission_df = pd.DataFrame(
        {
            "Patient_Week": patient_weeks,
            "FVC": fvc_preds,
            "Confidence": confidence_preds,
        }
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions generated: {len(submission_df)}")
    print("--- Inference Complete ---")
