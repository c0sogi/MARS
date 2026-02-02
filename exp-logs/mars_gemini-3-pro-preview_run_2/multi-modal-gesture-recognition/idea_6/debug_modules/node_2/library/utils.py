import torch
import numpy as np
import nltk
from library.config import Config, set_seed


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_levenshtein_distance(pred_seq, target_seq):
    """
    Computes the Levenshtein edit distance between two sequences.

    Args:
        pred_seq (list): Predicted sequence of labels.
        target_seq (list): Ground truth sequence of labels.

    Returns:
        int: Edit distance.
    """
    return nltk.edit_distance(pred_seq, target_seq)


def compute_competition_metric(predictions, ground_truths):
    """
    Computes the competition metric: Total Levenshtein Distance / Total True Gestures.

    Args:
        predictions (list of list of int): Predicted gesture sequences.
        ground_truths (list of list of int): Ground truth gesture sequences.

    Returns:
        float: The computed score (lower is better).
    """
    total_distance = 0
    total_gestures = 0

    for p, t in zip(predictions, ground_truths):
        total_distance += get_levenshtein_distance(p, t)
        total_gestures += len(t)

    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures


def compute_accuracy(logits, targets, ignore_index=None):
    """
    Computes frame-wise accuracy.

    Args:
        logits (torch.Tensor): Predictions of shape (B, C, T) or (N, C).
        targets (torch.Tensor): Targets of shape (B, T) or (N,).
        ignore_index (int, optional): Index to ignore in calculation.

    Returns:
        float: Accuracy value.
    """
    # Handle (B, C, T) shape commonly used in TCNs/CNNs
    if logits.dim() == 3:
        # Permute to (B, T, C) then flatten to (N, C)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
    elif logits.dim() == 2:
        # Assumes (N, C)
        pass

    preds = torch.argmax(logits, dim=1)

    if ignore_index is not None:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]

    if targets.numel() == 0:
        return 0.0

    correct = (preds == targets).sum().item()
    total = targets.numel()

    return correct / total


def apply_median_filter(probs, kernel_size=5):
    """
    Applies median filtering to predictions with nearest-neighbor padding.

    Args:
        probs (np.ndarray): Input array. Can be 1D (labels) or 2D (probabilities).
                            If 2D, argmax is applied first.
        kernel_size (int): Size of the median filter window.

    Returns:
        np.ndarray: Smoothed 1D label sequence.
    """
    if probs.ndim == 2:
        preds = np.argmax(probs, axis=-1)
    else:
        preds = probs.copy()

    # Nearest neighbor padding
    pad_size = kernel_size // 2
    preds_padded = np.pad(preds, (pad_size, pad_size), mode="edge")

    smoothed = np.zeros_like(preds)
    for i in range(len(preds)):
        window = preds_padded[i : i + kernel_size]
        smoothed[i] = np.median(window)

    return smoothed.astype(int)


def decode_predictions(frame_predictions):
    """
    Decodes frame-wise label predictions into a sequence of gesture IDs.
    Collapses repeated labels and removes background (class 0).

    Args:
        frame_predictions (np.ndarray or list): Sequence of frame labels.

    Returns:
        list of int: Decoded gesture sequence.
    """
    if len(frame_predictions) == 0:
        return []

    # Collapse repetitions
    collapsed = [frame_predictions[0]]
    for i in range(1, len(frame_predictions)):
        if frame_predictions[i] != frame_predictions[i - 1]:
            collapsed.append(frame_predictions[i])

    # Remove background (assuming 0 is background)
    final_sequence = [int(x) for x in collapsed if x != 0]

    return final_sequence
