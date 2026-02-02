import torch
import numpy as np
import pandas as pd
import os
from library.utils import set_seed


def predict_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Original, H-Flip, V-Flip).

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for the dataset (test or val).
        device (torch.device): Device to run inference on.

    Returns:
        dict: A dictionary mapping image IDs to predicted probabilities (0-1).
    """
    model.eval()
    model.to(device)
    results = {}

    with torch.no_grad():
        for batch in loader:
            # Handle different loader outputs (img, angle, id) or (img, angle, label, id)
            if len(batch) == 3:
                images, angles, ids = batch
            elif len(batch) == 4:
                images, angles, _, ids = batch
            else:
                raise ValueError(f"Unexpected batch structure with length {len(batch)}")

            images = images.to(device)
            angles = angles.to(device)

            # 1. Original View
            pred_1 = torch.sigmoid(model(images, angles))

            # 2. Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, dims=[3])
            pred_2 = torch.sigmoid(model(images_h, angles))

            # 3. Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, dims=[2])
            pred_3 = torch.sigmoid(model(images_v, angles))

            # Average predictions
            probs = (pred_1 + pred_2 + pred_3) / 3.0

            # Move to CPU and flatten
            probs_np = probs.cpu().numpy().flatten()

            # Store results
            for idx, img_id in enumerate(ids):
                results[img_id] = float(probs_np[idx])

    return results


def select_pseudo_labels(
    ensemble_predictions, confidence_threshold=0.95, variance_threshold=0.02
):
    """
    Selects pseudo-labels based on ensemble consensus and confidence.

    Args:
        ensemble_predictions (list of dict): List where each element is a result dict from predict_tta.
        confidence_threshold (float): Threshold for high confidence (p > thresh or p < 1-thresh).
        variance_threshold (float): Threshold for maximum standard deviation across ensemble.

    Returns:
        pd.DataFrame: DataFrame containing ['id', 'is_iceberg'] for selected samples.
    """
    if not ensemble_predictions:
        raise ValueError("ensemble_predictions list is empty")

    # Extract IDs from the first prediction dict
    ids = list(ensemble_predictions[0].keys())

    # Construct a DataFrame where each column is a model's prediction
    data = {"id": ids}
    for i, preds in enumerate(ensemble_predictions):
        # Ensure alignment of IDs
        try:
            col_values = [preds[uid] for uid in ids]
            data[f"model_{i}"] = col_values
        except KeyError:
            raise KeyError("Mismatch in IDs across ensemble predictions.")

    df = pd.DataFrame(data)

    # Calculate statistics
    pred_cols = [c for c in df.columns if c.startswith("model_")]
    df["mean_prob"] = df[pred_cols].mean(axis=1)
    df["std_prob"] = df[pred_cols].std(axis=1)

    # Define Criteria
    # 1. High Confidence: Close to 0 (Ship) or 1 (Iceberg)
    is_confident = (df["mean_prob"] > confidence_threshold) | (
        df["mean_prob"] < (1.0 - confidence_threshold)
    )

    # 2. Low Variance: Ensemble members agree
    is_stable = df["std_prob"] < variance_threshold

    # Filter
    selected_df = df[is_confident & is_stable].copy()

    # Assign Hard Labels
    # If prob > 0.5, it's an iceberg (1), else ship (0)
    selected_df["is_iceberg"] = (selected_df["mean_prob"] > 0.5).astype(float)

    print(f"Pseudo-Label Selection:")
    print(f"  Total Test Samples: {len(df)}")
    print(f"  Selected Samples:   {len(selected_df)}")
    print(f"  Selection Ratio:    {len(selected_df) / len(df):.4f}")

    return selected_df[["id", "is_iceberg"]]


def save_submission(predictions, output_path):
    """
    Saves predictions to a CSV file in the required format.

    Args:
        predictions (dict): Dictionary {id: probability}.
        output_path (str): Path to save the CSV.
    """
    df = pd.DataFrame(list(predictions.items()), columns=["id", "is_iceberg"])

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
