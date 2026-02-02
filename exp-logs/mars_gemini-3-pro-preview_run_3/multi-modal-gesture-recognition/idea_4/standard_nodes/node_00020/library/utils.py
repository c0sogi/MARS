import os
import csv
import torch
import numpy as np
from itertools import groupby
from library.config import Config

# Set fixed seeds for reproducibility
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


def decode_predictions(frame_predictions, min_len=5):
    """
    Decodes frame-wise predictions into a sequence of gesture IDs using Run-Length Encoding.
    Filters out background class and short segments.

    Args:
        frame_predictions (torch.Tensor or np.ndarray):
            Shape (Time, NumClasses) containing logits, or (Time,) containing class indices.
        min_len (int): Minimum duration (in frames) for a segment to be considered valid.

    Returns:
        list: Ordered list of gesture IDs (integers).
    """
    # Convert Tensor to Numpy if necessary
    if isinstance(frame_predictions, torch.Tensor):
        frame_predictions = frame_predictions.detach().cpu().numpy()

    # If input is logits (2D), apply argmax to get class indices
    if frame_predictions.ndim == 2:
        predictions = np.argmax(frame_predictions, axis=1)
    else:
        predictions = frame_predictions

    decoded_sequence = []

    # Run-Length Encoding using groupby
    # groupby returns consecutive keys and an iterator over the group
    for label, group in groupby(predictions):
        # Calculate the duration of the current segment
        duration = sum(1 for _ in group)

        # Filter:
        # 1. Ignore Background class (Index 0)
        # 2. Ignore segments shorter than min_len frames
        if label != 0 and duration >= min_len:
            decoded_sequence.append(int(label))

    return decoded_sequence


def write_submission_csv(predictions_dict, output_path=Config.SUBMISSION_FILE_PATH):
    """
    Writes the predictions to a CSV file in the required submission format.

    Format: SessionID,Label1,Label2,...

    Args:
        predictions_dict (dict): Mapping {sample_id (str): gesture_list (list of ints)}
        output_path (str): Path to save the CSV file.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sort by Sample ID to ensure deterministic output order
    sorted_sample_ids = sorted(predictions_dict.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        for sample_id in sorted_sample_ids:
            gestures = predictions_dict[sample_id]

            # Construct the row: [SessionID, Label1, Label2, ...]
            # Convert all gesture IDs to strings
            row = [sample_id] + [str(g) for g in gestures]

            writer.writerow(row)

    print(f"Submission saved to {output_path}")
