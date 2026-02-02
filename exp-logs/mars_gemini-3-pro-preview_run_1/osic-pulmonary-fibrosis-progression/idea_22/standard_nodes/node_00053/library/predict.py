import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything
from library.data import prepare_data
from library.model import CASDAN


def inference():
    """
    Executes the inference pipeline:
    1. Loads test data (images and metadata).
    2. Loads the trained CAS-DAN model.
    3. Predicts trajectory parameters (alpha, sigma_base, sigma_growth).
    4. Computes final FVC and Confidence using the parametric formula.
    5. Saves the results to submission.csv.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # prepare_data handles caching logic internally
    test_dataset = prepare_data("test", load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = CASDAN().to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading model weights from {Config.MODEL_SAVE_PATH}...")
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: No checkpoint found at {Config.MODEL_SAVE_PATH}. Using random initialization."
        )

    model.eval()

    # 4. Prediction Loop
    results = []
    print(f"Starting inference on {len(test_dataset)} samples...")

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            ax = batch["axial"].to(device)
            cor = batch["coronal"].to(device)
            tab = batch["tabular"].to(device)

            # Extract Metadata (Batched)
            # Note: In the Dataset, meta items are returned as tensors or lists
            base_fvc = batch["meta"]["Baseline_FVC"].to(device)
            base_week = batch["meta"]["Baseline_Week"].to(device)
            pred_week = batch["meta"]["Predict_Week"].to(device)
            patient_weeks = batch["meta"]["Patient_Week"]  # List of strings

            # Forward Pass
            # Model outputs trajectory parameters
            alpha, sigma_base, sigma_growth = model(ax, cor, tab)

            # --- Parametric Logic ---
            # Calculate time delta
            delta_t = pred_week - base_week

            # Predict FVC: Baseline + Slope * Delta_t
            fvc_pred = base_fvc + alpha * delta_t

            # Predict Confidence: Base_Conf + Growth_Conf * |Delta_t|
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_t)

            # Clip Confidence (Metric Requirement)
            sigma_pred = torch.clamp(sigma_pred, min=Config.CONFIDENCE_CLIP)

            # Convert to numpy for storage
            fvc_np = fvc_pred.cpu().numpy()
            sigma_np = sigma_pred.cpu().numpy()

            # Aggregate results
            for i in range(len(patient_weeks)):
                results.append(
                    {
                        "Patient_Week": patient_weeks[i],
                        "FVC": fvc_np[i],
                        "Confidence": sigma_np[i],
                    }
                )

    # 5. Save Submission
    sub_df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Generated {len(sub_df)} predictions.")
