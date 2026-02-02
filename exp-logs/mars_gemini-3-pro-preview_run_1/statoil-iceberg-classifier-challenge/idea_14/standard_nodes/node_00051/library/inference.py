import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import log_message


def predict_with_tta(model, loader, device=Config.DEVICE):
    """
    Generates predictions for a single model using Test Time Augmentation (TTA).
    TTA Strategy: Original, Horizontal Flip, Vertical Flip.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.

    Returns:
        dict: Mapping of id -> probability (float).
    """
    model.eval()
    model.to(device)

    results = {}

    with torch.no_grad():
        for images, angles, ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            # 1. Create TTA Views
            # images shape: (B, 3, 224, 224)
            # Horizontal Flip (dim 3)
            images_h = torch.flip(images, dims=[3])
            # Vertical Flip (dim 2)
            images_v = torch.flip(images, dims=[2])

            # 2. Forward Passes
            logits_orig = model(images, angles)
            logits_h = model(images_h, angles)
            logits_v = model(images_v, angles)

            # 3. Sigmoid & Average
            probs_orig = torch.sigmoid(logits_orig)
            probs_h = torch.sigmoid(logits_h)
            probs_v = torch.sigmoid(logits_v)

            # Shape: (B, 1)
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            # 4. Store
            batch_probs = avg_probs.cpu().numpy().flatten()

            for i, img_id in enumerate(ids):
                results[img_id] = float(batch_probs[i])

    return results


def predict_ensemble(models, loader, device=Config.DEVICE):
    """
    Generates predictions using an ensemble of models, each using TTA.

    Args:
        models (list): List of loaded PyTorch models.
        loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.

    Returns:
        dict: Mapping of id -> probability (float).
    """
    # Move all models to device and set to eval
    for model in models:
        model.eval()
        model.to(device)

    results = {}

    with torch.no_grad():
        for images, angles, ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            # TTA Views
            images_h = torch.flip(images, dims=[3])
            images_v = torch.flip(images, dims=[2])

            # Accumulator for ensemble probabilities
            ensemble_sum = None

            for model in models:
                # Forward pass for this model
                l_orig = model(images, angles)
                l_h = model(images_h, angles)
                l_v = model(images_v, angles)

                p_orig = torch.sigmoid(l_orig)
                p_h = torch.sigmoid(l_h)
                p_v = torch.sigmoid(l_v)

                # Average TTA for this model
                p_model_avg = (p_orig + p_h + p_v) / 3.0

                if ensemble_sum is None:
                    ensemble_sum = p_model_avg
                else:
                    ensemble_sum += p_model_avg

            # Average across models
            final_probs = ensemble_sum / len(models)

            # Store
            batch_probs = final_probs.cpu().numpy().flatten()

            for i, img_id in enumerate(ids):
                results[img_id] = float(batch_probs[i])

    return results


def create_submission(predictions, output_path=Config.SUBMISSION_PATH):
    """
    Generates the submission CSV file.

    Args:
        predictions (dict): Dictionary mapping id -> probability.
        output_path (str): Path to save the CSV.
    """
    # Load sample submission to ensure correct IDs and order
    if os.path.exists(Config.SAMPLE_SUBMISSION):
        df = pd.read_csv(Config.SAMPLE_SUBMISSION)

        # Map predictions to the DataFrame
        # We use map to ensure the probability corresponds to the correct ID
        df["is_iceberg"] = df["id"].map(predictions)

        # Fill missing values if any (though predictions should cover all)
        if df["is_iceberg"].isnull().any():
            missing_count = df["is_iceberg"].isnull().sum()
            log_message(
                f"Warning: {missing_count} IDs missing in predictions. Filling with 0.5."
            )
            df["is_iceberg"] = df["is_iceberg"].fillna(0.5)
    else:
        # Fallback: Create DataFrame directly from predictions
        log_message(
            "Sample submission not found. Creating submission from predictions dict."
        )
        df = pd.DataFrame(list(predictions.items()), columns=["id", "is_iceberg"])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    log_message(f"Submission saved to {output_path}")
