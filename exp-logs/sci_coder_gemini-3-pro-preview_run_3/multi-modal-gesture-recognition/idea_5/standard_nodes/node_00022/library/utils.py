import numpy as np
import nltk
from itertools import groupby
from library.config import Config


def decode_predictions(probabilities, threshold=5):
    """
    Decodes frame-wise probabilities or labels into a sequence of gesture IDs.

    This function applies the following steps:
    1. Converts probabilities to class indices (argmax) if necessary.
    2. Groups consecutive identical frames (Run-Length Encoding).
    3. Filters out the background class (Config.BACKGROUND_CLASS_ID).
    4. Filters out gesture segments shorter than the specified threshold.

    Args:
        probabilities (np.ndarray): Input array of shape (T, C) containing probabilities/logits,
                                    or shape (T,) containing integer class labels.
        threshold (int): Minimum duration (in frames) for a gesture segment to be considered valid.
                         Defaults to 5 as per instructions.

    Returns:
        List[int]: An ordered list of recognized gesture IDs.
    """
    # Handle both probabilities (T, C) and labels (T,)
    if probabilities.ndim == 2:
        labels = np.argmax(probabilities, axis=1)
    else:
        labels = probabilities

    predicted_gestures = []

    # Apply Run-Length Encoding (RLE) to collapse consecutive frames
    for label, group in groupby(labels):
        # Calculate the duration of the current segment
        length = sum(1 for _ in group)

        # Filter: Must not be background and must meet duration threshold
        if label != Config.BACKGROUND_CLASS_ID and length >= threshold:
            predicted_gestures.append(int(label))

    return predicted_gestures


def compute_levenshtein(predicted_sequences, ground_truth_sequences):
    """
    Computes the Levenshtein distance-based error rate metric for the challenge.

    The score is calculated as:
    Score = Sum(Levenshtein Distances) / Sum(Total Ground Truth Gestures)

    Args:
        predicted_sequences (List[List[int]]): A list where each element is a list of
                                               predicted gesture IDs for a sequence.
        ground_truth_sequences (List[List[int]]): A list where each element is a list of
                                                  ground truth gesture IDs for a sequence.

    Returns:
        float: The computed error rate (lower is better).
    """
    total_distance = 0
    total_gestures = 0

    for pred, truth in zip(predicted_sequences, ground_truth_sequences):
        # nltk.edit_distance handles lists of integers correctly
        dist = nltk.edit_distance(pred, truth)

        total_distance += dist
        total_gestures += len(truth)

    # Avoid division by zero if ground truth is empty (though unlikely in valid sets)
    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures
