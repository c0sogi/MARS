import torch
from library.utils import fbeta_score


@torch.no_grad()
def calculate_fbeta(logits, targets, beta=0.5, smooth=1e-6, threshold=0.5):
    """
    Calculates the F-beta score (specifically F0.5 for this competition) from model logits.

    This function acts as a wrapper around the library's fbeta_score. It handles the
    necessary conversion from raw model logits to probabilities using a sigmoid activation,
    which is required before applying the binary threshold.

    Args:
        logits (torch.Tensor): Raw output from the model (before sigmoid activation).
                               Shape: (N, C, H, W) or (N, 1, H, W).
        targets (torch.Tensor): Ground truth binary masks.
                                Shape: (N, C, H, W) or (N, 1, H, W).
        beta (float): The beta parameter for the F-score. Defaults to 0.5 to weight
                      precision higher than recall.
        smooth (float): Smoothing factor to avoid division by zero.
        threshold (float): The probability threshold to classify a pixel as ink (1)
                           vs background (0). Defaults to 0.5.

    Returns:
        float: The computed F-beta score.
    """
    # Convert raw logits to probabilities
    probs = torch.sigmoid(logits)

    # Calculate the score using the provided utility function
    # The fbeta_score function in library.utils handles thresholding and calculation
    score = fbeta_score(probs, targets, beta=beta, smooth=smooth, threshold=threshold)

    return score
