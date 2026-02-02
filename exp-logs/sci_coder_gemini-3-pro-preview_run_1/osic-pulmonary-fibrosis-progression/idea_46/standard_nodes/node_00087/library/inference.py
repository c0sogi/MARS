import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import SLHDANetwork


def predict_and_submit(
    model_path=Config.MODEL_SAVE_PATH,
    output_path=Config.SUBMISSION_PATH,
    device=Config.DEVICE,
    debug=Config.DEBUG,
):
    """
    Loads the trained SLH-DAN model, generates predictions for the test set
    using the parametric inference logic, and saves the submission CSV.

    Args:
        model_path (str): Path to the saved model checkpoint (.pth).
        output_path (str): Destination path for the submission CSV.
        device (str): Computation device ('cpu' or 'cuda').
        debug (bool): If True, uses a subset of data for quick testing.
    """
    # 1. Setup Environment
    Config.setup()
    seed_everything(Config.SEED)

    print(f"Inference Device: {device}")

    # 2. Load Data
    # We retrieve the test_loader. get_dataloaders returns (train, val, test).
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(debug=debug)

    # 3. Initialize Model
    print("Initializing SLH-DAN model architecture...")
    model = SLHDANetwork()
    model.to(device)

    # 4. Load Weights
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Model checkpoint not found at {model_path}. Predictions will be based on random initialization."
        )

    # 5. Inference Loop
    model.eval()
    results = []

    print("Starting inference...")
    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tab = batch["tabular"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            base_week = batch["base_week"].to(device)
            patient_ids = batch["patient_id"]

            # Forward pass
            # The model's forward method automatically handles the parametric inference:
            # It predicts alpha, sigma_base, sigma_growth and applies:
            # FVC = Base_FVC + alpha * (Week - Base_Week)
            # Confidence = Sigma_Base + Sigma_Growth * |Week - Base_Week|
            pred_fvc, pred_sigma = model(
                img_ax, img_cor, tab, weeks, base_fvc, base_week
            )

            # Move predictions to CPU
            pred_fvc_np = pred_fvc.cpu().numpy()
            pred_sigma_np = pred_sigma.cpu().numpy()
            weeks_np = weeks.cpu().numpy()

            # Aggregate results
            for i in range(len(patient_ids)):
                pid = patient_ids[i]
                wk = int(weeks_np[i])

                # Construct Patient_Week ID
                patient_week = f"{pid}_{wk}"

                results.append(
                    {
                        "Patient_Week": patient_week,
                        "FVC": pred_fvc_np[i],
                        "Confidence": pred_sigma_np[i],
                    }
                )

    # 6. Save Submission
    df = pd.DataFrame(results)

    # Ensure strict column ordering as per submission format
    df = df[["Patient_Week", "FVC", "Confidence"]]

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")
    print(f"Total predictions generated: {len(df)}")
