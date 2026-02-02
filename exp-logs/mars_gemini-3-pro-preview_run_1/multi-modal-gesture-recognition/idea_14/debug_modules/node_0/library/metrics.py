import numpy as np
import torch
import nltk
from scipy.signal import medfilt
from library.config import MEDIAN_FILTER_KERNEL, MIN_GESTURE_LENGTH, BACKGROUND_LABEL


def compute_levenshtein(predicted, target):
    """
    Calculates the Levenshtein distance between two sequences of gesture IDs.

    Args:
        predicted: List or array of predicted gesture IDs.
        target: List or array of ground truth gesture IDs.

    Returns:
        int: The edit distance.
    """
    # Ensure inputs are standard python lists for nltk
    if isinstance(predicted, (np.ndarray, torch.Tensor)):
        predicted = predicted.tolist()
    if isinstance(target, (np.ndarray, torch.Tensor)):
        target = target.tolist()

    return nltk.edit_distance(predicted, target)


def decode_predictions(logits, lengths):
    """
    Decodes frame-wise logits into a sequence of gesture IDs.
    Applies Median Filtering and Run-Length Encoding with filtering.

    Args:
        logits: (B, T, C) Tensor or Numpy array of raw scores.
        lengths: (B,) Tensor or Numpy array of valid sequence lengths.

    Returns:
        List[List[int]]: A list of predicted gesture sequences for each sample in the batch.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.detach().cpu().numpy()

    batch_size = logits.shape[0]
    predictions = []

    # Get class indices for every frame
    frame_preds = np.argmax(logits, axis=2)  # Shape: (B, T)

    for i in range(batch_size):
        length = int(lengths[i])

        # Handle empty sequences
        if length == 0:
            predictions.append([])
            continue

        # Slice valid frames for this sample
        raw_seq = frame_preds[i, :length]

        # Apply Median Filter to smooth noise
        # Kernel size must be odd for medfilt
        k = MEDIAN_FILTER_KERNEL
        if k % 2 == 0:
            k += 1

        # medfilt pads with 0 (which is BACKGROUND_LABEL), which is safe behavior here
        smoothed_seq = medfilt(raw_seq, kernel_size=k)

        # Run-Length Encoding & Filtering
        decoded_gestures = []

        if len(smoothed_seq) > 0:
            current_label = smoothed_seq[0]
            current_count = 0

            for t in range(len(smoothed_seq)):
                label = smoothed_seq[t]
                if label == current_label:
                    current_count += 1
                else:
                    # End of a contiguous segment
                    # Filter: Must not be background AND must meet min length
                    if (
                        current_label != BACKGROUND_LABEL
                        and current_count >= MIN_GESTURE_LENGTH
                    ):
                        decoded_gestures.append(int(current_label))

                    # Start new segment
                    current_label = label
                    current_count = 1

            # Handle the final segment
            if (
                current_label != BACKGROUND_LABEL
                and current_count >= MIN_GESTURE_LENGTH
            ):
                decoded_gestures.append(int(current_label))

        predictions.append(decoded_gestures)

    return predictions


def evaluate_batch(logits, lengths, targets):
    """
    Computes the total Levenshtein distance and total gesture count for a batch.

    Args:
        logits: (B, T, C) Model output scores.
        lengths: (B,) Valid sequence lengths.
        targets: List[List[int]] OR (B, L_max) Tensor padded with BACKGROUND_LABEL.

    Returns:
        total_dist: Sum of Levenshtein distances for the batch.
        total_len: Sum of lengths of ground truth sequences (denominator for error rate).
    """
    # Decode predictions
    preds = decode_predictions(logits, lengths)

    # Process targets into clean lists
    target_lists = []
    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
        for i in range(len(targets_np)):
            # Filter out background padding (0) to get the true sequence of gestures
            t_list = [int(x) for x in targets_np[i] if x != BACKGROUND_LABEL]
            target_lists.append(t_list)
    else:
        # Assume it's already a list of lists
        target_lists = targets

    total_dist = 0
    total_len = 0

    for p, t in zip(preds, target_lists):
        dist = compute_levenshtein(p, t)
        total_dist += dist
        total_len += len(t)

    return total_dist, total_len
