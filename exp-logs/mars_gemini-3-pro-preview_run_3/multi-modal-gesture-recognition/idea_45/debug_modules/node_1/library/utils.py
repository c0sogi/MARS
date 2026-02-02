import numpy as np
import pandas as pd
import os
import json
from itertools import groupby
from library import config


def levenshtein_distance(seq1, seq2):
    """
    Calculates the Levenshtein distance between two sequences.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1
                )
    return matrix[size_x - 1, size_y - 1]


def run_length_encoding(predictions):
    """
    Converts frame-wise predictions to a list of (label, duration) tuples.
    predictions: 1D array-like of class indices.
    """
    encoded = []
    for k, g in groupby(predictions):
        length = len(list(g))
        encoded.append((k, length))
    return encoded


def decode_predictions_to_labels(
    frame_predictions, min_length=config.MIN_GESTURE_LENGTH
):
    """
    Decodes frame-wise class indices into a list of gesture labels.
    1. Applies Run-Length Encoding.
    2. Filters out the background class (0).
    3. Filters out segments shorter than min_length.

    Returns: List of integer class IDs.
    """
    segments = run_length_encoding(frame_predictions)
    final_labels = []

    for label, duration in segments:
        # Skip background class
        if label == config.BACKGROUND_CLASS_ID:
            continue

        # Filter short segments
        if duration >= min_length:
            final_labels.append(int(label))

    return final_labels


def parse_ground_truth(metadata_df):
    """
    Parses the 'labels' column from the metadata DataFrame into a dictionary.
    Returns: {sample_id: [label_id, label_id, ...]}
    """
    gt_dict = {}
    for _, row in metadata_df.iterrows():
        sample_id = row["sample_id"]
        labels_json = row["labels"]

        if isinstance(labels_json, str):
            labels_list = json.loads(labels_json)
        else:
            labels_list = labels_json if labels_json is not None else []

        # Extract IDs in order
        # The metadata parsing logic already sorts by 'begin', so we just extract 'id'
        label_ids = [item["id"] for item in labels_list]
        gt_dict[sample_id] = label_ids

    return gt_dict


def compute_metric(predictions_dict, ground_truth_dict):
    """
    Computes the competition metric: Total Levenshtein Distance / Total GT Gestures.
    predictions_dict: {sample_id: [label_id, ...]}
    ground_truth_dict: {sample_id: [label_id, ...]}
    """
    total_distance = 0
    total_gt_gestures = 0

    # Iterate over all samples in ground truth
    for sample_id, gt_labels in ground_truth_dict.items():
        # Get predictions for this sample, default to empty list
        pred_labels = predictions_dict.get(sample_id, [])

        # Compute distance
        dist = levenshtein_distance(pred_labels, gt_labels)

        total_distance += dist
        total_gt_gestures += len(gt_labels)

    if total_gt_gestures == 0:
        return 0.0

    return total_distance / total_gt_gestures


def create_submission_file(predictions_dict, output_path):
    """
    Generates the submission CSV file in the format:
    SessionID,Label1,Label2,...

    predictions_dict: {sample_id: [label_id, ...]}
    output_path: Path to save the CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    # Sort by sample_id for consistent output
    for sample_id in sorted(predictions_dict.keys()):
        labels = predictions_dict[sample_id]

        # Join labels with commas
        if labels:
            label_str = ",".join(map(str, labels))
            row_str = f"{sample_id},{label_str}"
        else:
            # If no gestures predicted, just the ID
            row_str = f"{sample_id}"

        rows.append(row_str)

    with open(output_path, "w") as f:
        for row in rows:
            f.write(row + "\n")

    print(f"Submission file saved to {output_path}")
