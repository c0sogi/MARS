import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.model import GCRNet
from library.data import get_dataloaders

# Statistics used for normalization in data.py
# We need these to inverse transform the predictions
STATS = {
    "fvc_mean": 2654.65,
    "fvc_std": 801.70,
}


def generate_submission():
    """
    Loads the best model, performs inference on the test set,
    inverse transforms the predictions, and generates the submission CSV.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.get_device()
    print(f"Inference Device: {device}")

    # 2. Load Model
    print("Loading model...")
    model = GCRNet()
    model = model.to(device)

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Ensure training has run."
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Load Data
    # We only need the test loader
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(batch_size=Config.BATCH_SIZE)

    # 4. Inference
    print("Running inference...")
    patient_weeks = []
    fvc_preds = []
    conf_preds = []

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            p_weeks = batch["patient_week"]

            # Forward pass
            # Output shape: (Batch, 2) -> [Mu_scaled, Sigma_scaled]
            out = model(images, tabular)

            mu_scaled = out[:, 0]
            sigma_scaled = out[:, 1]

            # Inverse Transformation (Z-score inverse)
            # FVC = scaled * std + mean
            # Sigma = scaled * std (Scale only, no mean shift for standard deviation)
            mu_final = mu_scaled * STATS["fvc_std"] + STATS["fvc_mean"]
            sigma_final = sigma_scaled * STATS["fvc_std"]

            # Post-Processing: Clip Confidence
            # The metric clips sigma at 70ml. We enforce this lower bound.
            sigma_final = torch.max(sigma_final, torch.tensor(70.0).to(device))

            # Store results
            patient_weeks.extend(p_weeks)
            fvc_preds.extend(mu_final.cpu().numpy())
            conf_preds.extend(sigma_final.cpu().numpy())

    # 5. Create Submission DataFrame
    submission = pd.DataFrame(
        {"Patient_Week": patient_weeks, "FVC": fvc_preds, "Confidence": conf_preds}
    )

    # Ensure correct data types (FVC is typically int in submission, but float is usually accepted.
    # Sample submission shows integers for FVC and Confidence)
    # However, keeping as float is safer for precision, but let's match sample format if possible.
    # The sample submission provided in description shows ints.
    # We will round FVC and Confidence to nearest integer for the final file.
    # Note: The metric uses float logic, but CSV format often implies specific types.
    # Given the prompt sample: "2000,100", let's cast to int.
    # Actually, let's keep them as is or simple formatting to avoid losing precision if the evaluation allows floats.
    # Looking at sample: "2000,100".
    # Let's round to be safe and match the visual style of the sample.
    submission["FVC"] = submission["FVC"].round().astype(int)
    submission["Confidence"] = submission["Confidence"].round().astype(int)

    # 6. Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_FILE, index=False)

    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(submission.head())
