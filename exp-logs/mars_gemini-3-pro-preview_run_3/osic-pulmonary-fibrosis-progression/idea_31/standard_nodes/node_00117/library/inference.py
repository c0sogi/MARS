import os
import torch
import pandas as pd
from library.config import Config
from library.model import DSPRNet
from library.data import get_dataloaders
from library.utils import seed_everything, InverseScaler


def generate_submission(debug=False):
    """
    Generates the submission file for the test set using the trained ZIMARNet model.

    Args:
        debug (bool): If True, runs on a subset of data for debugging purposes.
    """
    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing inference on {device}...")

    # 2. Prepare Data
    # get_dataloaders handles metadata loading, image caching, and batching.
    # We only need the test_loader.
    _, _, test_loader = get_dataloaders(debug=debug)

    # 3. Load Model
    model = DSPRNet().to(device)

    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading model checkpoint from {Config.BEST_MODEL_PATH}...")
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Checkpoint not found at {Config.BEST_MODEL_PATH}. Using random initialization."
        )

    model.eval()

    # 4. Initialize Scaler
    # The scaler loads training statistics to invert the Z-score normalization
    scaler = InverseScaler()

    # 5. Inference Loop
    results = []

    print("Starting prediction loop...")
    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            imgs = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            patient_weeks = batch["patient_week"]

            # Forward pass
            # Model returns scaled Mu and scaled Sigma (softplus applied)
            mu_scaled, sigma_scaled = model(imgs, clinical)

            # Move to CPU for post-processing
            mu_scaled = mu_scaled.cpu()
            sigma_scaled = sigma_scaled.cpu()

            # Inverse Transformation
            # Convert Z-scored FVC and scaled Sigma back to milliliters
            mu_raw, sigma_raw = scaler(mu_scaled, sigma_scaled)

            # Post-processing Constraints
            # Metric requires sigma to be clipped at 70ml
            sigma_final = torch.clamp(sigma_raw, min=Config.SIGMA_MIN)

            # Collect results
            for pw, fvc, conf in zip(patient_weeks, mu_raw, sigma_final):
                results.append(
                    {"Patient_Week": pw, "FVC": fvc.item(), "Confidence": conf.item()}
                )

    # 6. Format and Save Submission
    submission = pd.DataFrame(results)

    # Ensure correct column order as per sample_submission.csv
    submission = submission[["Patient_Week", "FVC", "Confidence"]]

    # Save to disk
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions generated: {len(submission)}")
