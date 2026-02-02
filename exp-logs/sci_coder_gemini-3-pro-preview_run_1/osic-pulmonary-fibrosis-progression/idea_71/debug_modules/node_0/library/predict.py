import os
import torch
import numpy as np
import pandas as pd

from library.config import Config, seed_everything
from library.data import get_test_dataloader
from library.model import DBSLNet


def inference_step(device_name=Config.DEVICE):
    """
    Loads the best model and performs inference on the test set.

    Returns:
        tuple: (fvc_preds, sigma_preds) as numpy arrays.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(device_name)

    # 2. Load Model
    model = DBSLNet()
    model.to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Model loaded successfully from {Config.MODEL_SAVE_PATH}")
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.MODEL_SAVE_PATH}. Using random initialization."
        )

    model.eval()

    # 3. Load Data
    # get_test_dataloader reads Config.TEST_CSV internally
    test_loader = get_test_dataloader()

    all_fvc = []
    all_sigma = []

    print("Starting inference on test set...")

    # 4. Inference Loop
    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            week = batch["week"].to(device)
            base_week = batch["base_week"].to(device)
            base_fvc = batch["base_fvc"].to(device)

            # Forward pass
            # The model internally handles the parametric logic:
            # FVC = Base + Alpha * dt
            # Sigma = Base_Sigma + Growth_Sigma * |dt|
            fvc_pred, sigma_pred = model(
                img_ax, img_cor, tabular, week, base_week, base_fvc
            )

            # Collect results
            all_fvc.append(fvc_pred.cpu().numpy())
            all_sigma.append(sigma_pred.cpu().numpy())

    # Concatenate all batches
    fvc_preds = np.concatenate(all_fvc)
    sigma_preds = np.concatenate(all_sigma)

    return fvc_preds, sigma_preds


def generate_submission(fvc_preds, sigma_preds):
    """
    Formats the predictions into the submission format and saves to CSV.

    Args:
        fvc_preds (np.array): Predicted FVC values.
        sigma_preds (np.array): Predicted Confidence values.
    """
    # Load metadata to get the correct Patient_Week identifiers
    # The loader is not shuffled, so the order matches Config.TEST_CSV
    test_df = pd.read_csv(Config.TEST_CSV)

    if len(test_df) != len(fvc_preds):
        raise ValueError(
            f"Shape mismatch: Metadata has {len(test_df)} rows, predictions have {len(fvc_preds)} rows."
        )

    # Clip Confidence values at 70 ml as per metric requirements
    sigma_clipped = np.maximum(sigma_preds, 70.0)

    # Construct Submission DataFrame
    submission = pd.DataFrame(
        {
            "Patient_Week": test_df["Patient_Week"],
            "FVC": fvc_preds,
            "Confidence": sigma_clipped,
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Sample of generated submission:")
    print(submission.head())


def run_prediction():
    """
    Main entry point to run the prediction pipeline.
    """
    fvc, sigma = inference_step()
    generate_submission(fvc, sigma)
