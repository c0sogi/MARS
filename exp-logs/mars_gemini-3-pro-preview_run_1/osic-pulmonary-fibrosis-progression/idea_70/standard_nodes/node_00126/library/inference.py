import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import PGBBNet
from library.data import get_test_loader
from library.utils import load_checkpoint


def generate_submission(
    checkpoint_path=os.path.join(Config.WORKING_DIR, "best_model.pth"),
    output_file="./submission/submission.csv",
):
    """
    Generates the submission file for the test set using the trained PGBBNet model.

    Args:
        checkpoint_path (str): Path to the trained model checkpoint.
        output_file (str): Path where the submission CSV will be saved.
    """
    device = torch.device(Config.DEVICE)

    # 1. Initialize Model
    model = PGBBNet().to(device)

    # 2. Load Weights
    if not os.path.exists(checkpoint_path):
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Predictions will be random."
        )
    else:
        print(f"Loading model from {checkpoint_path}...")
        load_checkpoint(checkpoint_path, model, device=device)

    model.eval()

    # 3. Get Test Loader
    # The loader handles image caching internally via LungDataset
    test_loader = get_test_loader()

    results = []

    print("Starting inference on test set...")

    # 4. Inference Loop
    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)

            patient_weeks = batch["patient_week"]

            # Forward pass
            # Model returns (B, 2) -> [FVC, Confidence]
            preds = model(axial, coronal, tabular, meta)

            preds_np = preds.cpu().numpy()

            # Collect results
            for i, pw in enumerate(patient_weeks):
                fvc = preds_np[i, 0]
                confidence = preds_np[i, 1]

                results.append(
                    {"Patient_Week": pw, "FVC": fvc, "Confidence": confidence}
                )

    # 5. Create DataFrame
    submission_df = pd.DataFrame(results)

    # 6. Post-processing
    # Ensure Confidence is at least 70 as per metric description (approximate measurement uncertainty)
    # Although the metric function handles this, it is safer to clip in submission.
    submission_df["Confidence"] = submission_df["Confidence"].clip(lower=70.0)

    # 7. Save Submission
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    print(f"Total predictions: {len(submission_df)}")
    print(submission_df.head())
