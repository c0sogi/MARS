import torch
import numpy as np
from library.config import DEVICE


def predict(model, loader, device=DEVICE):
    """
    Performs inference on a dataset using a trained model with Test Time Augmentation (TTA).

    This function iterates through the provided DataLoader, computing predictions for both
    the original images and their horizontally flipped counterparts. The probabilities
    are averaged to produce the final prediction. It returns both the predicted probabilities
    and the corresponding targets (which are ground truth labels for validation sets or
    image IDs for test sets).

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        loader (torch.utils.data.DataLoader): DataLoader containing the images.
        device (str): The device to run inference on ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing two numpy arrays:
            - final_probs: Array of predicted probabilities for class 1 (dog).
            - final_targets: Array of corresponding targets (labels or IDs).
    """
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            # --- 1. Inference on Original Images ---
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # --- 2. Inference on Flipped Images (TTA) ---
            # Flip images horizontally. Assuming NCHW format, width is dim 3.
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip)
            probs_flip = torch.sigmoid(logits_flip)

            # --- 3. Average Probabilities ---
            probs_avg = (probs_orig + probs_flip) / 2.0

            # Store results
            # Flatten to ensure we get a 1D array of probabilities
            all_probs.append(probs_avg.cpu().numpy().flatten())

            # Handle targets:
            # For validation, targets are float tensors (labels).
            # For test, targets are int tensors (IDs) via default collate.
            if isinstance(targets, torch.Tensor):
                all_targets.append(targets.cpu().numpy().flatten())
            else:
                all_targets.append(np.array(targets).flatten())

    # Concatenate results from all batches
    if len(all_probs) > 0:
        final_probs = np.concatenate(all_probs)
        final_targets = np.concatenate(all_targets)
    else:
        final_probs = np.array([])
        final_targets = np.array([])

    return final_probs, final_targets
