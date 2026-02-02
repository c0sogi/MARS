import os
import torch
import pandas as pd
import numpy as np
from library import config, utils


def predict_with_tta(model, loader, device):
    """
    Generates predictions for the test set using Test Time Augmentation (TTA).
    Specifically, it averages the predictions of the original image and a
    horizontally flipped version to improve stability.

    Args:
        model (torch.nn.Module): The trained model (in eval mode).
        loader (torch.utils.data.DataLoader): The test data loader.
        device (str): The device to run inference on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary mapping image ID (int) to predicted probability (float).
    """
    model.eval()
    predictions = {}

    # Ensure no gradients are calculated during inference
    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # --- Pass 1: Original Images ---
            outputs_orig = model(images)
            # Apply Sigmoid to get probabilities (BCEWithLogitsLoss output is logits)
            probs_orig = torch.sigmoid(outputs_orig)

            # --- Pass 2: TTA (Horizontal Flip) ---
            # Flip images along the width dimension (dim 3 for NCHW format)
            images_flipped = torch.flip(images, dims=[3])
            outputs_flipped = model(images_flipped)
            probs_flipped = torch.sigmoid(outputs_flipped)

            # --- Aggregate ---
            # Arithmetic mean of the two passes
            avg_probs = (probs_orig + probs_flipped) / 2.0

            # Move to CPU and flatten
            batch_probs = avg_probs.cpu().numpy().flatten()
            batch_ids = ids.numpy().flatten()

            # Store results mapped by ID
            for img_id, prob in zip(batch_ids, batch_probs):
                predictions[int(img_id)] = float(prob)

    return predictions


def average_ensemble_predictions(prediction_dicts):
    """
    Computes the arithmetic mean of predictions from multiple models.

    Args:
        prediction_dicts (list of dict): A list where each element is a dictionary
                                         mapping image IDs to probabilities (output of predict_with_tta).

    Returns:
        dict: A dictionary mapping image ID to the averaged probability.
    """
    if not prediction_dicts:
        return {}

    # Initialize result dictionary with keys from the first model's predictions
    avg_preds = {k: 0.0 for k in prediction_dicts[0].keys()}
    num_models = len(prediction_dicts)

    # Sum probabilities across all models
    for img_id in avg_preds:
        total_prob = 0.0
        for preds in prediction_dicts:
            # Use .get() for safety, though keys should be identical
            total_prob += preds.get(img_id, 0.0)

        # Calculate mean
        avg_preds[img_id] = total_prob / num_models

    return avg_preds


def generate_submission(
    predictions, output_dir="./submission", filename="submission.csv"
):
    """
    Formats the predictions and saves them to a CSV file in the required submission format.

    Args:
        predictions (dict): Dictionary mapping image ID to probability.
        output_dir (str): Directory to save the submission file.
        filename (str): Name of the submission file.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Sort IDs to ensure consistent ordering (1, 2, 3...)
    sorted_ids = sorted(predictions.keys())
    sorted_probs = [predictions[i] for i in sorted_ids]

    # Create DataFrame
    df = pd.DataFrame({"id": sorted_ids, "label": sorted_probs})

    # Save to CSV without the index
    df.to_csv(filepath, index=False)
    print(f"Submission saved to {filepath}")
