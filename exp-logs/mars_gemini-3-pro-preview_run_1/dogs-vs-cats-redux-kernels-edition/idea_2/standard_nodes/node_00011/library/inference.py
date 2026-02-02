import os
import torch
import numpy as np
from torch.cuda.amp import autocast
from library.config import Config
from library.utils import save_submission


def predict_with_tta(model, data_loader, device):
    """
    Generates predictions for the test set using Test Time Augmentation (TTA).

    Strategy:
    1. Predict on the original image.
    2. Predict on the horizontally flipped image.
    3. Average the probabilities (not logits) of both views.

    Args:
        model (torch.nn.Module): The trained model in evaluation mode.
        data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device (CPU/GPU) to perform inference on.

    Returns:
        tuple: (ids, probabilities)
            - ids (np.ndarray): Array of image IDs.
            - probabilities (np.ndarray): Array of predicted probabilities for the 'dog' class.
    """
    model.eval()

    all_ids = []
    all_probs = []

    with torch.no_grad():
        for images, ids in data_loader:
            images = images.to(device)

            # Collect IDs (handle tensor batching from DataLoader)
            if isinstance(ids, torch.Tensor):
                all_ids.extend(ids.cpu().numpy())
            else:
                all_ids.extend(ids)

            with autocast():
                # --- Forward Pass 1: Original Image ---
                logits_orig = model(images)
                probs_orig = torch.sigmoid(logits_orig)

                # --- Forward Pass 2: Horizontally Flipped Image ---
                # Input shape is (N, C, H, W). Flip along width axis (dim=3).
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped)

                # --- Aggregate ---
                # Average the probabilities
                avg_probs = (probs_orig + probs_flipped) / 2.0

            # Store batch predictions
            all_probs.extend(avg_probs.cpu().numpy())

    return np.array(all_ids), np.array(all_probs)


def run_inference_and_save(model, data_loader, device, output_path=None):
    """
    Orchestrates the inference process: runs TTA predictions and saves the submission file.

    Args:
        model (torch.nn.Module): The loaded model.
        data_loader (torch.utils.data.DataLoader): Test data loader.
        device (torch.device): Device to run on.
        output_path (str, optional): Full path to save the submission CSV.
                                     Defaults to the path defined in Config.
    """
    if output_path is None:
        output_path = os.path.join(Config.submission_dir, "submission.csv")

    print(f"Starting inference with TTA on device: {device}...")

    # Generate predictions
    ids, probs = predict_with_tta(model, data_loader, device)

    # Save to disk
    print(f"Inference complete. Saving {len(ids)} predictions to {output_path}...")
    save_submission(ids, probs, output_path=output_path)
    print("Submission saved successfully.")
