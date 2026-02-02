import os
import torch
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import PyramidSymmetryDifferenceModel


def predict_and_submit(debug=Config.DEBUG, load_cached_data=True):
    """
    Generates predictions for the test set using the trained Pyramid Symmetry-Difference model.
    Aggregates predictions by prediction_id (taking the max probability across views) and
    saves the result to submission.csv.

    Args:
        debug (bool): If True, runs on a subset of data.
        load_cached_data (bool): If True, attempts to load processed metadata from cache.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Inference Device: {device}")

    # 2. Data Loading
    # We unpack the tuple to get only the test_loader
    print("Loading Test Data...")
    _, _, test_loader = get_dataloaders(debug=debug, load_cached_data=load_cached_data)

    # 3. Model Initialization
    print("Initializing Model...")
    model = PyramidSymmetryDifferenceModel().to(device)

    # 4. Load Weights
    weights_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(
            f"Warning: Weights file not found at {weights_path}. Using random initialization."
        )

    model.eval()

    # 5. Inference Loop
    print("Starting Inference...")
    results = []

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            img_target = batch["image"].to(device)
            img_contra = batch["contra_image"].to(device)
            prediction_ids = batch[Config.ID_COL]

            # Forward pass
            logits = model(img_target, img_contra)

            # Apply Sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Collect results
            for pid, prob in zip(prediction_ids, probs):
                results.append({Config.ID_COL: pid, Config.TARGET_COL: prob})

    # 6. Aggregation
    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    if df_results.empty:
        print("Warning: No predictions generated. Creating empty submission.")
        submission = pd.DataFrame(columns=[Config.ID_COL, Config.TARGET_COL])
    else:
        # Group by prediction_id and take the MAX probability across views (CC, MLO)
        # This handles the requirement: "Multiple images will share the same prediction ID."
        submission = (
            df_results.groupby(Config.ID_COL)[Config.TARGET_COL].max().reset_index()
        )

    # 7. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Submission Head:")
    print(submission.head())

    return submission
