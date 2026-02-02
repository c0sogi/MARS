import os
import cv2
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility by wrapping Config.set_seed.

    Args:
        seed (int): The seed value to use.
    """
    Config.set_seed(seed)


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) as per competition metric.

    The metric checks that pairs are sorted, positive, and decoded pixel values are not duplicated.
    Pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

    Args:
        mask (np.ndarray): Binary mask (0 or 1), shape (H, W).

    Returns:
        str: Space-delimited list of pairs (start, length).
    """
    # Flatten the mask in row-major order (C-style) to match the "left to right, top to bottom" numbering
    pixels = mask.flatten()

    # Prepend and append 0 to efficiently detect transitions between 0 and 1
    # This handles cases where the mask starts or ends with ink (1)
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is the start of the first ink run
    # runs[1] is the start of the first background run (end of first ink run)
    # The length of the ink run is runs[1] - runs[0]
    # We replace the end indices with lengths
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def fbeta_score(preds, targets, beta=0.5, smooth=1e-6, threshold=0.5):
    """
    Calculates the F-beta score for binary segmentation.

    Args:
        preds (torch.Tensor): Predicted probabilities or logits.
        targets (torch.Tensor): Ground truth binary mask.
        beta (float): Beta value for F-score (default 0.5 weights precision higher).
        smooth (float): Smoothing factor to avoid division by zero.
        threshold (float): Threshold to binarize predictions.

    Returns:
        float: F-beta score.
    """
    # Apply threshold to get binary predictions
    preds_bin = (preds > threshold).float().view(-1)
    targets_bin = targets.float().view(-1)

    tp = (preds_bin * targets_bin).sum()
    fp = (preds_bin * (1 - targets_bin).float()).sum()
    fn = ((1 - preds_bin) * targets_bin).sum()

    beta_sq = beta**2

    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = (numerator + smooth) / (denominator + smooth)
    return score.item()


def create_submission(predictions, output_path):
    """
    Creates and saves the submission CSV file.

    Args:
        predictions (dict): Dictionary mapping fragment_id (str) to RLE string (str).
        output_path (str): Path to save the submission.csv file.
    """
    ids = []
    rles = []

    # Sort by ID to keep it tidy, though not strictly required
    for frag_id in sorted(predictions.keys()):
        ids.append(frag_id)
        rles.append(predictions[frag_id])

    df = pd.DataFrame({"Id": ids, "Predicted": rles})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)


def load_image(path, grayscale=False):
    """
    Loads an image from the specified path.

    Args:
        path (str): Path to the image file.
        grayscale (bool): If True, loads as grayscale (cv2.IMREAD_GRAYSCALE).
                          Otherwise loads unchanged (cv2.IMREAD_UNCHANGED).

    Returns:
        np.ndarray or None: The loaded image, or None if path does not exist.
    """
    if not os.path.exists(path):
        return None

    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    return cv2.imread(path, flags)
