import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import PGARNet
from library.data import get_dataloaders


def generate_submission(load_cached_data=True, batch_size=None):
    """
    Generates the submission file for the Lung Function Decline prediction task.

    Args:
        load_cached_data (bool): If True, attempts to use cached preprocessed images.
                                 If False, or if cache is missing, processes from scratch.
        batch_size (int, optional): Override default batch size for inference.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    # 1. Setup
    device = Config.DEVICE
    print(f"Inference device: {device}")

    # Update batch size in Config if provided
    if batch_size is not None:
        Config.BATCH_SIZE = batch_size

    # 2. Load Data
    # We rely on get_dataloaders to handle caching and preprocessing via library.data
    # We only need the test_loader
    print(f"Loading test data (Cached: {load_cached_data})...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # Load Metadata to get Patient_Week IDs
    # The test_loader is built from Config.TEST_CSV, so the order is guaranteed to match
    test_meta_df = pd.read_csv(Config.TEST_CSV)
    patient_weeks = test_meta_df["Patient_Week"].values

    print(f"Total test samples to predict: {len(patient_weeks)}")

    # 3. Load Model
    model = PGARNet()
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. Please train the model first."
        )

    print(f"Loading model weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference Loop
    fvc_preds = []
    sigma_preds = []

    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            # Move inputs to device
            axial = inputs["axial"].to(device)
            coronal = inputs["coronal"].to(device)
            tabular = inputs["tabular"].to(device)
            dt = inputs["dt"].to(device)
            base_fvc = inputs["base_fvc"].to(device)

            # Forward pass
            # When dt and base_fvc are provided, the model returns the calculated trajectory points
            # fvc_pred = base + alpha * dt
            # sigma_pred = base_sigma + growth_sigma * |dt|
            fvc, sigma = model(axial, coronal, tabular, dt, base_fvc)

            # Collect results
            # Flatten in case of batch dimension surprises, though shape should be (B,)
            fvc_preds.extend(fvc.cpu().numpy().flatten().tolist())
            sigma_preds.extend(sigma.cpu().numpy().flatten().tolist())

    # 5. Assemble Submission
    if len(fvc_preds) != len(patient_weeks):
        raise ValueError(
            f"Prediction mismatch: Generated {len(fvc_preds)} predictions for {len(patient_weeks)} IDs."
        )

    submission_df = pd.DataFrame(
        {"Patient_Week": patient_weeks, "FVC": fvc_preds, "Confidence": sigma_preds}
    )

    # 6. Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)

    print(f"Submission successfully saved to: {save_path}")
    print("First 5 rows:")
    print(submission_df.head())

    return submission_df
