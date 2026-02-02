import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from library.config import SUBMISSION_PATH, DEVICE
from library.trainer import evaluate
from library.model import FocalLoss


def find_best_threshold(val_probs, val_targets):
    """
    Finds the optimal probability threshold that maximizes the sample-averaged F1 score.
    Instead of a fixed range, this function uses the percentiles of the predicted
    probabilities to dynamically define the search space, adapting to the model's
    confidence distribution.

    Args:
        val_probs (np.ndarray): Predicted probabilities for validation set (n_samples, n_classes).
        val_targets (np.ndarray): Binary ground truth labels (n_samples, n_classes).

    Returns:
        best_thresh (float): The threshold value that yielded the highest F1 score.
        best_score (float): The maximum F1 score achieved.
    """
    # Flatten probabilities to analyze the global distribution of scores
    flat_probs = val_probs.flatten()

    # Calculate percentiles to define a relevant search range.
    # Since this is a multi-label problem with sparse targets, valid positive predictions
    # often lie in the upper tail of the probability distribution.
    # We search between the 95th percentile and the 99.9th percentile.
    p_min = np.percentile(flat_probs, 95)
    p_max = np.percentile(flat_probs, 99.9)

    # Handle edge cases where model outputs are extremely uniform
    if p_max <= p_min:
        p_min = 0.1
        p_max = 0.9

    # Generate candidate thresholds: 50 steps within the percentile range
    thresholds = np.linspace(p_min, p_max, 50)

    # Ensure 0.5 is always checked as a baseline
    thresholds = np.unique(np.concatenate([thresholds, [0.5]]))

    # Filter thresholds to be strictly within (0, 1)
    thresholds = thresholds[(thresholds > 0) & (thresholds < 1)]

    best_thresh = 0.5
    best_score = -1.0

    print(
        f"Optimizing threshold within percentile-based range: [{thresholds.min():.6f}, {thresholds.max():.6f}]"
    )

    for t in thresholds:
        preds = (val_probs > t).astype(int)
        # Calculate Mean F1-Score (samples average)
        score = f1_score(val_targets, preds, average="samples", zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = t

    print(f"Optimal Threshold Found: {best_thresh:.6f} with Val F1: {best_score:.16f}")
    return best_thresh, best_score


def generate_submission(
    model, test_loader, test_ids, preprocessor, threshold, device=DEVICE
):
    """
    Generates predictions for the test set using the trained model and optimized threshold.
    Decodes binary predictions back to tag strings and saves the submission file.

    Args:
        model (torch.nn.Module): The trained WideAndDeepModel.
        test_loader (DataLoader): DataLoader for the test set.
        test_ids (np.ndarray): Array of Question IDs corresponding to the test set.
        preprocessor (TextPreprocessor): The fitted preprocessor containing the MultiLabelBinarizer.
        threshold (float): The optimal probability threshold for classification.
        device (torch.device): The device to run inference on.

    Returns:
        submission_df (pd.DataFrame): The generated submission dataframe.
    """
    print("Generating predictions for test set...")

    # Instantiate loss function required by evaluate() signature, though unused for inference
    criterion = FocalLoss()

    # Run inference
    # evaluate returns: avg_loss, all_probs, all_targets
    _, test_probs, _ = evaluate(model, test_loader, criterion, device)

    print(f"Applying threshold: {threshold:.6f}")
    # Apply threshold to get binary predictions
    test_preds_binary = (test_probs > threshold).astype(int)

    print("Decoding binary predictions to tags...")
    # Inverse transform using the MultiLabelBinarizer from preprocessor
    # mlb.inverse_transform returns a list of tuples, e.g., [('c#', 'java'), ('php',)]
    test_tags_tuples = preprocessor.mlb.inverse_transform(test_preds_binary)

    # Join tags with space to match submission format
    test_tags_str = [" ".join(tags) for tags in test_tags_tuples]

    # Create DataFrame
    submission_df = pd.DataFrame({"Id": test_ids, "Tags": test_tags_str})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Submission generated successfully.")

    return submission_df
