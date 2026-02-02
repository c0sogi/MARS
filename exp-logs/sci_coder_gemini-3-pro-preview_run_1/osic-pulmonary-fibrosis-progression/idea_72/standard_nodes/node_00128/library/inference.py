import torch
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.data import get_test_dataloader
from library.model import AASLNet


def predict(debug=False, limit_batches=None):
    """
    Runs the inference pipeline to generate predictions for the test set.

    Args:
        debug (bool): If True, runs in debug mode (currently unused but kept for interface consistency).
        limit_batches (int, optional): If provided, limits the number of batches processed.
                                       Useful for quick debugging of the inference flow.

    Returns:
        pd.DataFrame: The submission dataframe containing Patient_Week, FVC, and Confidence.
    """
    # Setup device
    device = torch.device(Config.DEVICE)
    print(f"Inference running on device: {device}")

    # Load Data
    # The loader reads from ./metadata/test.csv and handles image processing/caching
    test_loader = get_test_dataloader()

    # Initialize Model
    model = AASLNet()
    model.to(device)

    # Load Weights
    # We check if the weight file exists. If not (e.g. initial run), we warn but proceed
    # to ensure the pipeline code is valid.
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading model weights from {Config.MODEL_SAVE_PATH}...")
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Model weights not found at {Config.MODEL_SAVE_PATH}.")
        print("Proceeding with random initialization for testing purposes.")

    # Set model to evaluation mode
    model.eval()

    results = []

    print("Starting prediction loop...")

    with torch.no_grad():
        for batch_idx, (img_ax, img_cor, tabular, meta, patient_week_ids) in enumerate(
            test_loader
        ):
            # Check debug limit
            if limit_batches is not None and batch_idx >= limit_batches:
                print(
                    f"Debug limit reached ({limit_batches} batches). Stopping inference."
                )
                break

            # Move data to device
            img_ax = img_ax.to(device)
            img_cor = img_cor.to(device)
            tabular = tabular.to(device)
            meta = meta.to(device)

            # Forward pass
            # AASLNet.forward() returns the projected FVC and Sigma for the specific week
            # defined in the 'meta' tensor.
            pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, meta)

            # Move predictions to CPU and convert to numpy
            pred_fvc = pred_fvc.cpu().numpy()
            pred_sigma = pred_sigma.cpu().numpy()

            # Process batch results
            for i in range(len(patient_week_ids)):
                p_week = patient_week_ids[i]
                fvc = float(pred_fvc[i])
                sigma = float(pred_sigma[i])

                # Clip Confidence
                # The metric clips confidence at 70ml. We enforce this lower bound
                # in the submission to reflect the approximate measurement uncertainty.
                sigma = max(sigma, Config.MIN_CONFIDENCE)

                results.append(
                    {"Patient_Week": p_week, "FVC": fvc, "Confidence": sigma}
                )

    # Create Submission DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure columns are in the correct order required by the competition
    submission_df = submission_df[["Patient_Week", "FVC", "Confidence"]]

    # Save to disk
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Inference complete.")
    print(f"Generated predictions for {len(submission_df)} rows.")
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")

    return submission_df
