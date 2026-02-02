import numpy as np
import torch


def calculate_fbeta(pred_mask, true_mask, beta=0.5, epsilon=1e-7):
    """
    Calculates the F-beta score for binary segmentation masks.

    The F-beta score is a weighted harmonic mean of precision and recall.
    F0.5 weights precision higher than recall.

    Args:
        pred_mask (np.ndarray): Binary predicted mask (0 or 1).
        true_mask (np.ndarray): Binary ground truth mask (0 or 1).
        beta (float): The beta parameter for the F-score. Defaults to 0.5.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The calculated F-beta score.
    """
    # Ensure inputs are flat numpy arrays
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.detach().cpu().numpy()

    p = pred_mask.reshape(-1)
    t = true_mask.reshape(-1)

    # Calculate basic confusion matrix stats
    # TP: Predicted 1 and True 1
    tp = np.sum(p * t)
    # Total Predicted Positives: TP + FP
    pred_positives = np.sum(p)
    # Total True Positives: TP + FN
    true_positives = np.sum(t)

    fp = pred_positives - tp
    fn = true_positives - tp

    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp + epsilon)
    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn + epsilon)

    # F-beta calculation
    beta_sq = beta**2
    f_beta = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + epsilon)
    )

    return float(f_beta)


def rle_encode(mask):
    """
    Run-Length Encode a binary mask.

    The metric checks that the pairs are sorted, positive, and the decoded pixel
    values are not duplicated. The pixels are numbered from left to right,
    then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited run-length encoding string (e.g., '1 3 10 5').
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Flatten in row-major order (C-style) as per "left to right, then top to bottom"
    pixels = mask.flatten()

    # We prepend and append 0 to detect runs that start at the first pixel
    # or end at the last pixel.
    # pixels[1:] != pixels[:-1] detects transitions.
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs array now contains indices where value changes.
    # Because we padded with 0 at the start:
    # Even indices in 'runs' (0, 2, 4...) correspond to 0->1 transitions (Starts)
    # Odd indices in 'runs' (1, 3, 5...) correspond to 1->0 transitions (Ends)

    # Note: The competition uses 1-based indexing.
    # Our 'runs' indices are already effectively 1-based relative to the original unpadded array
    # because of the shift introduced by np.where on the padded array.
    # Example: [0, 1, 1, 0] -> padded [0, 0, 1, 1, 0, 0]
    # Diff at indices 1 (0->1) and 3 (1->0).
    # runs = [2, 4].
    # Start at pixel 2 (1-based index for first '1'). Length = 4-2 = 2.

    starts = runs[0::2]
    ends = runs[1::2]
    lengths = ends - starts

    # Interleave starts and lengths
    encoding = []
    for s, l in zip(starts, lengths):
        encoding.append(str(s))
        encoding.append(str(l))

    return " ".join(encoding)


def optimize_threshold(val_probs, val_labels, beta=0.5, steps=100):
    """
    Finds the optimal probability threshold that maximizes the F-beta score
    on the validation set.

    Args:
        val_probs (np.ndarray): Predicted probabilities (0.0 to 1.0).
        val_labels (np.ndarray): Ground truth labels (0 or 1).
        beta (float): Beta parameter for F-score.
        steps (int): Number of threshold steps to check between 0.01 and 0.99.

    Returns:
        tuple: (best_threshold, best_f_score)
    """
    if isinstance(val_probs, torch.Tensor):
        val_probs = val_probs.detach().cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.detach().cpu().numpy()

    # Flatten once to speed up loop
    probs_flat = val_probs.reshape(-1)
    labels_flat = val_labels.reshape(-1)

    best_threshold = 0.5
    best_score = 0.0

    # Search space
    thresholds = np.linspace(0.01, 0.99, steps)

    for thresh in thresholds:
        # Binarize predictions
        preds_bin = (probs_flat >= thresh).astype(np.uint8)

        # Calculate score
        score = calculate_fbeta(preds_bin, labels_flat, beta=beta)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score
