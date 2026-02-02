import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger(name="inference")


def predict_with_tta(model, dataloader, device=Config.DEVICE):
    """
    Performs inference on the test set using Test Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The trained model to use for inference.
        dataloader (DataLoader): The test data loader.
        device (torch.device): The device to run inference on.

    Returns:
        tuple: (ids, probabilities)
            ids (list): List of image IDs corresponding to the predictions.
            probabilities (np.ndarray): Array of shape (N_samples, N_classes) containing averaged probabilities.
    """
    model.eval()
    all_probs = []
    all_ids = []

    logger.info(f"Starting inference with TTA on {len(dataloader.dataset)} samples...")

    with torch.no_grad():
        for i, (images, ids) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)

            # 1. Forward pass: Original Images
            logits_orig = model(images)
            probs_orig = F.softmax(logits_orig, dim=1)

            # 2. Forward pass: Flipped Images (TTA)
            # Flip along the width dimension (dim 3 for NCHW format)
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped)
            probs_flipped = F.softmax(logits_flipped, dim=1)

            # 3. Average probabilities
            avg_probs = (probs_orig + probs_flipped) / 2.0

            # Move to CPU and store
            all_probs.append(avg_probs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all batches
    final_probs = np.concatenate(all_probs, axis=0)

    logger.info("Inference completed.")
    return all_ids, final_probs


def save_submission(ids, probs, class_names, output_path=Config.SUBMISSION_PATH):
    """
    Formats and saves the predictions to a CSV file.

    Args:
        ids (list): List of image IDs.
        probs (np.ndarray): Prediction probabilities matrix.
        class_names (list or np.ndarray): List of class names for column headers.
        output_path (str): File path to save the submission CSV.
    """
    logger.info(f"Preparing submission file at {output_path}...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame(probs, columns=class_names)

    # Insert ID column at the beginning
    df.insert(0, "id", ids)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Submission saved successfully with shape {df.shape}.")
