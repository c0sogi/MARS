import os
import numpy as np
from scipy.ndimage import median_filter
from itertools import groupby
from library.config import Config


def apply_median_filter(labels, kernel_size=15):
    """
    Applies a median filter to the discrete label sequence to smooth out noise.
    Uses nearest-neighbor padding to protect valid gestures at the sequence boundaries,
    preventing the erosion of start/end gestures.

    Args:
        labels (np.ndarray): 1D array of discrete class labels (T,).
        kernel_size (int): The size of the median filter window.
                           Defaults to 15 (approx. 0.75s - 1.5s depending on FPS).

    Returns:
        np.ndarray: The smoothed label sequence.
    """
    # mode='nearest' repeats the edge values, ensuring that gestures starting
    # or ending at the very beginning/end of the video are not filtered out
    # as noise by zero-padding.
    return median_filter(labels, size=kernel_size, mode="nearest")


def decode_predictions(labels):
    """
    Decodes a sequence of frame-wise labels into a list of gesture IDs.
    1. Collapses consecutive repeated labels (e.g., [2, 2, 2, 0, 0] -> [2, 0]).
    2. Removes the background class (Class 0).

    Args:
        labels (np.ndarray or list): Sequence of frame-wise labels.

    Returns:
        list: Ordered list of recognized gesture IDs (integers).
    """
    # Collapse consecutive duplicates using itertools.groupby
    # groupby returns (key, group_iterator), we only need the key
    collapsed = [key for key, _ in groupby(labels)]

    # Remove background class (0)
    # Based on Config, Class 0 is Background
    decoded = [int(x) for x in collapsed if x != 0]

    return decoded


def generate_submission(predictions, output_filename="submission.csv", kernel_size=15):
    """
    Processes model predictions and saves them to a CSV file in the required format.
    Format: SessionID,gesture_1,gesture_2,...

    Args:
        predictions (dict): Dictionary mapping sample_id (str) to either:
                            - Probabilities: np.ndarray of shape (T, NumClasses)
                            - Labels: np.ndarray of shape (T,)
        output_filename (str): Name of the output file to save in Config.SUBMISSION_DIR.
        kernel_size (int): Window size for the median filter.
    """
    output_path = os.path.join(Config.SUBMISSION_DIR, output_filename)

    lines = []

    # Sort sample IDs to ensure deterministic output order
    sample_ids = sorted(predictions.keys())

    print(f"Generating submission for {len(sample_ids)} sequences...")

    for sample_id in sample_ids:
        pred_data = predictions[sample_id]

        # Convert probabilities to discrete labels if necessary
        if pred_data.ndim == 2:
            # pred_data is (T, C), take argmax along class dimension
            labels = np.argmax(pred_data, axis=1)
        else:
            # pred_data is already (T,) labels
            labels = pred_data

        # 1. Apply Median Filter (Label-Space Smoothing)
        smoothed_labels = apply_median_filter(labels, kernel_size=kernel_size)

        # 2. Decode (Collapse & Remove Background)
        gesture_ids = decode_predictions(smoothed_labels)

        # 3. Format the output line
        # Format: SessionID,label1,label2,label3
        if not gesture_ids:
            # If no gestures are detected, just output the SessionID
            line = f"{sample_id}"
        else:
            gestures_str = ",".join(map(str, gesture_ids))
            line = f"{sample_id},{gestures_str}"

        lines.append(line)

    # Write to file
    with open(output_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Submission saved to {output_path}")
