import os
import torch
import pandas as pd
import numpy as np
from library.utils import get_device, seed_everything
from library.dataset import get_dataloaders
from library.model import BCSLNet


def predict_test_loader(model, loader, device, limit_batches=None):
    """
    Generates predictions for the test set using the trained model.

    Args:
        model (nn.Module): The trained BCSLNet model.
        loader (DataLoader): The test data loader.
        device (torch.device): The computation device.
        limit_batches (int, optional): Limit the number of batches for debugging.

    Returns:
        pd.DataFrame: DataFrame containing Patient_Week, FVC, and Confidence.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if limit_batches is not None and i >= limit_batches:
                break

            # Move inputs to device
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            delta_week = batch["delta_week"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            patient_weeks = batch["patient_week"]

            # Forward pass
            # The model returns FVC and Sigma based on the parametric trajectory logic:
            # FVC = Base + Alpha * Delta_Week
            # Sigma = Base_Sigma + Growth_Sigma * |Delta_Week|
            fvc_pred, sigma_pred = model(axial, coronal, tabular, delta_week, base_fvc)

            # Move results to CPU
            fvc_pred = fvc_pred.cpu().numpy()
            sigma_pred = sigma_pred.cpu().numpy()

            # Process batch results
            for pw, f, s in zip(patient_weeks, fvc_pred, sigma_pred):
                # Clip confidence to a minimum of 70 ml as per metric requirements
                # to reflect approximate measurement uncertainty.
                s_clipped = max(float(s), 70.0)

                results.append(
                    {"Patient_Week": pw, "FVC": float(f), "Confidence": s_clipped}
                )

    return pd.DataFrame(results)


def generate_submission(
    model_path="./working/best_model.pth",
    output_path="./submission/submission.csv",
    batch_size=16,
    limit_batches=None,
):
    """
    Loads the model, generates predictions for the test set, and saves the submission file.

    Args:
        model_path (str): Path to the saved model weights.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        limit_batches (int, optional): Limit number of batches for debugging.
    """
    # Ensure reproducibility
    seed_everything(42)

    device = get_device()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Retrieve Test Loader (index 2)
    # We ignore train and val loaders here
    _, _, test_loader = get_dataloaders(batch_size=batch_size)

    # Initialize Model Architecture
    model = BCSLNet().to(device)

    # Load Model Weights
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model weights from {model_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return
    else:
        print(
            f"Warning: Model file {model_path} not found. Predictions will be based on random initialization."
        )

    # Generate Predictions
    print("Generating predictions...")
    df_results = predict_test_loader(
        model, test_loader, device, limit_batches=limit_batches
    )

    # Save Submission
    df_results.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df_results)} rows.")
