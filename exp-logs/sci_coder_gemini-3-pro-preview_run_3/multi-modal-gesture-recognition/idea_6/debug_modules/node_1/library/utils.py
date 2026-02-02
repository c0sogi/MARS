import os
import random
import numpy as np
import torch
import pandas as pd
from nltk.metrics.distance import edit_distance
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
    """

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


def decode_predictions(probabilities, threshold=5):
    """
    Decodes frame-wise class probabilities into a sequence of gesture IDs.

    Logic:
    1. Argmax to get frame-wise labels.
    2. Run-Length Encoding (RLE) to group consecutive identical labels.
    3. Filter out the background class (ID 0).
    4. Filter out segments shorter than the threshold duration.

    Args:
        probabilities (torch.Tensor or np.ndarray): Shape (T, NumClasses).
        threshold (int): Minimum duration (in frames) for a gesture to be considered valid.

    Returns:
        list: A list of integer gesture IDs.
    """
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    # 1. Argmax
    pred_labels = np.argmax(probabilities, axis=1)

    if len(pred_labels) == 0:
        return []

    # 2. Run-Length Encoding
    segments = []
    current_label = pred_labels[0]
    current_count = 1

    for i in range(1, len(pred_labels)):
        if pred_labels[i] == current_label:
            current_count += 1
        else:
            segments.append((current_label, current_count))
            current_label = pred_labels[i]
            current_count = 1
    segments.append((current_label, current_count))

    # 3. & 4. Filter Background and Short Segments
    final_gestures = []
    for label, count in segments:
        if label != 0 and count >= threshold:
            final_gestures.append(int(label))

    return final_gestures


def compute_levenshtein(predicted_seq, target_seq):
    """
    Computes the Levenshtein (Edit) Distance between two sequences of gesture IDs.

    Args:
        predicted_seq (list): List of predicted gesture IDs.
        target_seq (list): List of ground truth gesture IDs.

    Returns:
        int: The edit distance.
    """
    return edit_distance(predicted_seq, target_seq)


def compute_dataset_score(predictions_dict, targets_dict):
    """
    Computes the global error rate for the dataset.
    Metric = Sum(Levenshtein Distances) / Sum(Total Ground Truth Gestures)

    Args:
        predictions_dict (dict): Mapping {sample_id: [predicted_labels]}
        targets_dict (dict): Mapping {sample_id: [ground_truth_labels]}

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_gestures = 0

    for sample_id, target_seq in targets_dict.items():
        # Get prediction for this sample, default to empty if missing
        pred_seq = predictions_dict.get(sample_id, [])

        # Compute distance
        dist = compute_levenshtein(pred_seq, target_seq)

        total_distance += dist
        total_gestures += len(target_seq)

    if total_gestures == 0:
        return 0.0 if total_distance == 0 else float("inf")

    return total_distance / total_gestures


def save_submission(predictions_dict, output_path):
    """
    Saves predictions to a CSV file in the required submission format.
    Format: SessionID,Label1,Label2,...

    Args:
        predictions_dict (dict): Mapping {sample_id: [predicted_labels]}
        output_path (str): Path to save the CSV file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    # Sort by sample_id to ensure consistent order
    sorted_ids = sorted(predictions_dict.keys())

    for sample_id in sorted_ids:
        labels = predictions_dict[sample_id]
        # Convert list of ints to comma-separated string
        labels_str = ",".join(map(str, labels))
        rows.append(f"{sample_id},{labels_str}")

    with open(output_path, "w") as f:
        for row in rows:
            f.write(row + "\n")

    print(f"Submission saved to {output_path}")
