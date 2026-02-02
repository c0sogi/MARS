import os
import torch
import pandas as pd
import numpy as np
from library.config import DEVICE, SUBMISSION_DIR, CACHE_DIR, seed_everything
from library.data import get_dataloaders
from library.model import HybridEfficientNet


def predict_and_submit(
    model_path: str = os.path.join(CACHE_DIR, "best_model.pth"),
    output_path: str = os.path.join(SUBMISSION_DIR, "submission.csv"),
    debug: bool = False,
    load_cached_data: bool = True,
):
    """
    Loads the trained model, performs inference on the test set, aggregates
    predictions by prediction_id, and saves the submission file.

    Args:
        model_path (str): Path to the saved model state dictionary.
        output_path (str): Path where the submission CSV will be saved.
        debug (bool): If True, runs inference on a subset of data for testing.
        load_cached_data (bool): Whether to use cached metadata processing.
    """
    # Ensure reproducibility
    seed_everything()

    print(f"Initializing inference pipeline on device: {DEVICE}")

    # 1. Load Data
    # We retrieve the test_loader and the number of tabular features needed for model init.
    # get_dataloaders handles metadata loading and processing internally.
    _, _, test_loader, num_tabular_features = get_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    # 2. Initialize Model
    # We set pretrained=False because we are loading our own trained weights.
    model = HybridEfficientNet(
        num_tabular_features=num_tabular_features,
        backbone_name="efficientnet_b0",
        pretrained=False,
    )

    # Check if model weights exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    # Load weights
    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 3. Inference Loop
    all_probs = []
    all_ids = []

    print("Starting inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            # The test dataset returns ((images, tabular), prediction_ids)
            (images, tabular), pred_ids = batch

            images = images.to(DEVICE)
            tabular = tabular.to(DEVICE)

            # Forward pass
            logits = model((images, tabular))

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_ids.extend(pred_ids)

    # 4. Aggregation
    # Create a DataFrame with image-level predictions
    df_pred = pd.DataFrame({"prediction_id": all_ids, "cancer": all_probs})

    # The task requires one prediction per prediction_id.
    # Since there are multiple images (views) per breast, we take the max probability.
    print("Aggregating predictions by prediction_id (max pooling)...")
    submission_df = df_pred.groupby("prediction_id", as_index=False)["cancer"].max()

    # 5. Save Submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(f"Total unique predictions: {len(submission_df)}")

    # Print first few rows for verification
    print("Sample predictions:")
    print(submission_df.head())

    return submission_df
