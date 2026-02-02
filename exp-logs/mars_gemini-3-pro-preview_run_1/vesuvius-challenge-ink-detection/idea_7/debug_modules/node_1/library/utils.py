import numpy as np
import torch
import pandas as pd
import os
import random
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_fbeta(y_true, y_pred, beta=0.5, epsilon=1e-7):
    """
    Calculates the F-beta score, which is the weighted harmonic mean of precision and recall.

    Args:
        y_true: Ground truth binary mask (Tensor or numpy array).
        y_pred: Predicted binary mask (Tensor or numpy array).
        beta: The beta parameter for F-beta score (default 0.5).
        epsilon: Small constant to prevent division by zero.

    Returns:
        float: The F-beta score.
    """
    # Convert tensors to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten the arrays to 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate F-beta score
    # Formula: (1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)
    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + epsilon)

    return float(score)


def rle_encode(img):
    """
    Run-length encodes a binary mask according to the competition format.
    Pixels are numbered from left to right, then top to bottom (row-major).

    Args:
        img: Binary mask (2D numpy array), where 1 indicates ink and 0 indicates background.

    Returns:
        str: Space-delimited run-length encoded string (e.g., '1 3 10 5').
    """
    # Flatten the image in row-major order (default 'C')
    pixels = img.flatten()

    # We need to find where the value changes.
    # We pad with 0 at the beginning and end to detect transitions at edges.
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    # np.where returns indices in the padded array.
    # Because of the leading 0 pad, the index in 'runs' corresponds directly
    # to the 1-based index in the original flattened array for the start of the change.
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The output format requires pairs of (start_position, run_length).
    # 'runs' currently holds the start indices of alternating 1s and 0s.
    # Since we padded with 0, the first change (if any) must be 0->1 (start of ink).
    # So runs[::2] are the start positions of ink.
    # runs[1::2] are the start positions of the next non-ink region (end of ink).

    if len(runs) % 2 != 0:
        # This should technically not happen with 0-padding on both sides,
        # but as a safeguard for malformed inputs.
        runs = np.append(runs, len(pixels) - 1)

    # Calculate lengths: end_pos - start_pos
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def optimize_threshold(y_true, y_pred_probs, beta=0.5, steps=100):
    """
    Finds the optimal probability threshold that maximizes the F-beta score on the validation set.

    Args:
        y_true: Ground truth binary labels (numpy array or Tensor).
        y_pred_probs: Predicted probabilities (numpy array or Tensor), values between 0 and 1.
        beta: Beta parameter for F-score.
        steps: Number of threshold steps to evaluate (default 100).

    Returns:
        float: The optimal threshold value.
    """
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred_probs):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()

    y_true = y_true.flatten()
    y_pred_probs = y_pred_probs.flatten()

    best_threshold = 0.5
    best_score = 0.0

    # Search space: 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, steps)

    for thr in thresholds:
        # Binarize predictions based on current threshold
        y_pred = (y_pred_probs >= thr).astype(np.uint8)

        # Calculate metric
        score = calculate_fbeta(y_true, y_pred, beta=beta)

        if score > best_score:
            best_score = score
            best_threshold = thr

    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best F{beta} Score: {best_score}")  # Printing full precision as requested

    return best_threshold


def write_submission(predictions, output_path=config.SUBMISSION_PATH):
    """
    Writes the final submission CSV file.

    Args:
        predictions: Dictionary mapping fragment_id (str) to RLE string (str).
        output_path: Path to save the CSV file.
    """
    ids = []
    rles = []

    # Sort by ID to ensure consistent order (though not strictly required, it's good practice)
    for frag_id in sorted(predictions.keys()):
        ids.append(frag_id)
        rles.append(predictions[frag_id])

    df = pd.DataFrame({"Id": ids, "Predicted": rles})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
