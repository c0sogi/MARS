import os
import torch
import numpy as np
import pandas as pd

from library.config import DEVICE, SEED, setup_reproducibility

# Import existing functions from library.trainer to satisfy the requirement
# of not re-implementing provided logic.
from library.trainer import optimize_threshold, generate_submission

# Ensure reproducibility for inference steps
setup_reproducibility(SEED)


def predict_probs(model, loader, device=DEVICE):
    """
    Generates raw probability predictions for the provided data loader.

    This function iterates through the loader, performs a forward pass
    using the model in evaluation mode, and aggregates the results.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        loader (torch.utils.data.DataLoader): DataLoader containing the dataset (e.g., test set).
        device (str): The device to perform inference on (default: configured DEVICE).

    Returns:
        np.ndarray: A 1D numpy array containing the predicted probabilities for each sample.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # Handle the unpacking of the batch.
            # The ContactDataset returns a tuple (X, y).
            if isinstance(batch, (list, tuple)):
                X = batch[0]
            else:
                X = batch

            # Move inputs to the appropriate device
            X = X.to(device)

            # Forward pass
            outputs = model(X)

            # Collect predictions (move to CPU and convert to numpy)
            all_preds.append(outputs.cpu().numpy())

    if not all_preds:
        return np.array([])

    # Concatenate all batch outputs (shape: [N, 1]) and flatten to 1D array (shape: [N,])
    return np.concatenate(all_preds).flatten()
