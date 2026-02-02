import os
import torch
import numpy as np
import pandas as pd
from library import config


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions using Test Time Augmentation (TTA).
    Predicts on Original, Horizontally Flipped, and Vertically Flipped images.

    Args:
        model (nn.Module): The trained PyTorch model.
        dataloader (DataLoader): DataLoader containing the test dataset.
        device (str): Device to perform inference on ('cuda' or 'cpu').

    Returns:
        np.ndarray: A 1D numpy array of predicted probabilities.
    """
    model.eval()
    all_probs = []

    # Ensure no gradients are computed
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # 1. Original View
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip View (Flip width dimension: dim 3 in NCHW)
            images_h = torch.flip(images, dims=[3])
            logits_h = model(images_h)
            probs_h = torch.sigmoid(logits_h)

            # 3. Vertical Flip View (Flip height dimension: dim 2 in NCHW)
            images_v = torch.flip(images, dims=[2])
            logits_v = model(images_v)
            probs_v = torch.sigmoid(logits_v)

            # Average the probabilities from all views
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            all_probs.append(avg_probs.cpu().numpy())

    # Concatenate all batches and flatten to 1D array
    # Model output is (B, 1), so concatenate gives (N, 1), flatten gives (N,)
    return np.concatenate(all_probs).flatten()


def save_submission(ids, probs, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (np.ndarray or list): List of image IDs (filenames).
        probs (np.ndarray or list): List of predicted probabilities.
        output_path (str): Full path where the CSV should be saved.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame matching the sample_submission format
    df = pd.DataFrame({"id": ids, "has_cactus": probs})

    # Save to CSV without index
    df.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")
