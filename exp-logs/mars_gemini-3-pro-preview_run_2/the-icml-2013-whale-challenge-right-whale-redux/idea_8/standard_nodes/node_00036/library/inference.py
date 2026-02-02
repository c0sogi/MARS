import os
import torch
import numpy as np
import pandas as pd
from library.config import Config


def predict(model, loader, device):
    """
    Generates predictions for the test set using a trained model.
    Runs a forward pass without gradients and applies sigmoid activation.

    Args:
        model (torch.nn.Module): The trained model instance.
        loader (torch.utils.data.DataLoader): The test dataloader yielding (data, clips).
        device (torch.device): The device (CPU/GPU) to run inference on.

    Returns:
        tuple: (clips, probabilities)
            clips (np.ndarray): Array of clip filenames.
            probabilities (np.ndarray): Array of predicted probabilities (0.0 to 1.0).
    """
    model.eval()
    all_clips = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            # Test loader yields (data, clips)
            if len(batch) != 2:
                raise ValueError(
                    "DataLoader must yield (data, clips) tuples for inference."
                )

            data, clips = batch
            data = data.to(device)

            # Forward pass
            logits = model(data)

            # Apply Sigmoid activation to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_clips.extend(clips)
            all_probs.extend(probs)

    return np.array(all_clips), np.array(all_probs)


def save_submission(clips, probabilities, output_path=Config.SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the competition format.

    Args:
        clips (np.ndarray): Array of clip filenames.
        probabilities (np.ndarray): Array of predicted probabilities.
        output_path (str): File path to save the CSV.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    submission = pd.DataFrame({"clip": clips, "probability": probabilities})

    # Save to CSV
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission.head())


def run_inference(models, loader, device, output_path=Config.SUBMISSION_PATH):
    """
    Manages the prediction process for an ensemble of models.
    Aggregates predictions via soft voting (averaging) and saves the result.

    Args:
        models (list[torch.nn.Module]): List of trained model instances.
        loader (torch.utils.data.DataLoader): The test dataloader.
        device (torch.device): The device to run inference on.
        output_path (str): Path to save the final submission file.
    """
    print(f"Starting inference with {len(models)} models...")

    accumulated_probs = None
    reference_clips = None

    for i, model in enumerate(models):
        print(f"Running prediction for model {i + 1}/{len(models)}...")
        clips, probs = predict(model, loader, device)

        if reference_clips is None:
            reference_clips = clips
            accumulated_probs = probs
        else:
            # Ensure clip order is identical across models
            if not np.array_equal(reference_clips, clips):
                raise ValueError(
                    f"Clip mismatch detected for model {i + 1}. Dataloader order must be deterministic."
                )

            accumulated_probs += probs

    # Compute average probabilities (Soft Voting)
    avg_probs = accumulated_probs / len(models)

    # Save final submission
    save_submission(reference_clips, avg_probs, output_path)
