import os
import torch
import pandas as pd
import numpy as np

from library.config import DEVICE, WORKING_DIR, SUBMISSION_PATH, SEED
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import PyramidSiameseEfficientNet


def generate_predictions(load_cached_data=True, max_samples=None):
    """
    Generates predictions for the test set using the best trained model.
    Aggregates predictions by prediction_id (taking the max probability across views)
    and saves the result to submission.csv.

    Args:
        load_cached_data (bool): Whether to use cached metadata/stats.
        max_samples (int, optional): Limit the number of samples for debugging.
    """
    # 1. Reproducibility
    seed_everything(SEED)

    # 2. Data Loading
    # We retrieve the test_loader. get_dataloaders handles the Siamese pairing
    # and preprocessing (including age/implant channel construction).
    print("Initializing DataLoaders for inference...")
    _, _, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, max_samples=max_samples
    )

    # 3. Model Initialization
    print("Initializing Pyramid Siamese Network...")
    model = PyramidSiameseEfficientNet()
    model = model.to(DEVICE)

    # 4. Load Weights
    model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model file {model_path} not found! Predictions will be based on random initialization."
        )

    # 5. Inference Loop
    model.eval()

    prediction_ids = []
    probabilities = []

    print("Running inference on test set...")
    with torch.no_grad():
        # Iterate over test loader
        # Format from dataset: target_img, contra_img, label (dummy), pred_id
        for target_img, contra_img, _, pred_ids in test_loader:
            target_img = target_img.to(DEVICE)
            contra_img = contra_img.to(DEVICE)

            # Forward pass through Siamese Network
            logits = model(target_img, contra_img)

            # Apply Sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Store results
            prediction_ids.extend(pred_ids)
            probabilities.extend(probs)

    # 6. Aggregation
    # Create DataFrame from raw image-level predictions
    df_pred = pd.DataFrame({"prediction_id": prediction_ids, "cancer": probabilities})

    # Group by prediction_id and take Max probability
    # This aggregates multiple views (e.g., CC and MLO) for the same breast into a single prediction.
    # If a view suggests high probability of cancer, the breast is flagged.
    df_submission = df_pred.groupby("prediction_id")["cancer"].max().reset_index()

    # 7. Save Submission
    print(f"Saving submission to {SUBMISSION_PATH}...")
    df_submission.to_csv(SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")

    # Print head for verification
    print(df_submission.head())

    return df_submission
