import os
import torch
import numpy as np
import pandas as pd
from library.config import Config


def predict_test_set(model, loader, device, tta=Config.TTA_FLIP):
    """
    Performs inference on the test set (dataset mode='test').

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.
        tta (bool): Whether to use Test-Time Augmentation (Horizontal Flip).

    Returns:
        ids (np.ndarray): Array of recording IDs.
        probs (np.ndarray): Array of predicted probabilities (N_samples, N_classes).
    """
    model.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, rec_ids in loader:
            images = images.to(device)

            # Forward pass 1: Original
            logits = model(images)
            probs = torch.sigmoid(logits)

            if tta:
                # Forward pass 2: Horizontal Flip
                # images shape: (B, C, H, W). Flip on W (dim 3).
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped)

                # Average probabilities
                probs = (probs + probs_flipped) / 2.0

            all_probs.append(probs.cpu().numpy())
            all_ids.append(rec_ids.numpy())

    if len(all_probs) == 0:
        return np.array([]), np.array([])

    ids = np.concatenate(all_ids)
    probs = np.concatenate(all_probs)

    return ids, probs


def predict_ensemble(models, loader, device, tta=Config.TTA_FLIP):
    """
    Performs inference using an ensemble of models and averages the probabilities.

    Args:
        models (list): List of trained models.
        loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.
        tta (bool): Whether to use TTA.

    Returns:
        ids (np.ndarray): Array of recording IDs.
        probs (np.ndarray): Averaged probabilities.
    """
    avg_probs = None
    ids = None

    for i, model in enumerate(models):
        curr_ids, curr_probs = predict_test_set(model, loader, device, tta=tta)

        if i == 0:
            ids = curr_ids
            avg_probs = curr_probs
        else:
            # Assuming loader order is deterministic and consistent
            avg_probs += curr_probs

    if avg_probs is not None:
        avg_probs /= len(models)

    return ids, avg_probs


def save_pseudo_labels(ids, probs, output_path):
    """
    Saves predictions as pseudo-labels in Parquet format.

    Args:
        ids (np.ndarray): Recording IDs.
        probs (np.ndarray): Predicted probabilities.
        output_path (str): Path to save the parquet file.
    """
    data = {"rec_id": ids}
    for i in range(probs.shape[1]):
        data[f"species_{i}"] = probs[:, i]

    df = pd.DataFrame(data)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_parquet(output_path, index=False)
    print(f"Pseudo-labels saved to {output_path}")


def create_submission(ids, probs, output_path):
    """
    Generates the submission CSV file.

    Args:
        ids (np.ndarray): Recording IDs.
        probs (np.ndarray): Predicted probabilities.
        output_path (str): Path to save the submission CSV.
    """
    submission_rows = []

    # Iterate through each sample
    for idx, rec_id in enumerate(ids):
        sample_probs = probs[idx]
        # Iterate through each species
        for species_idx, prob in enumerate(sample_probs):
            # Construct Id: rec_id * 100 + species_idx
            submission_id = int(rec_id * 100 + species_idx)
            submission_rows.append({"Id": submission_id, "Probability": prob})

    df_submission = pd.DataFrame(submission_rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
