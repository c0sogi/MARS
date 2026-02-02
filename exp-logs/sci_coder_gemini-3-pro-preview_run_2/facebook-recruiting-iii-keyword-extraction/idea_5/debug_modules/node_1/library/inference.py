import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from library.config import Config
from library.utils import Timer, set_seed


def find_best_threshold(y_true, y_probs):
    """
    Finds the optimal probability threshold for multi-label classification
    by maximizing the Mean F1-Score (Samples).

    Uses Distribution-Aware Threshold Tuning based on percentiles of the
    predicted probabilities to define the search range.

    Args:
        y_true (np.ndarray): Binary target matrix of shape (N, C).
        y_probs (np.ndarray): Predicted probability matrix of shape (N, C).

    Returns:
        float: The optimal threshold value.
    """
    print("Finding optimal threshold...")

    # Flatten probabilities to analyze distribution
    # Using a random sample if the data is too large to sort efficiently
    flat_probs = y_probs.ravel()
    if len(flat_probs) > 10_000_000:
        sample_probs = np.random.choice(flat_probs, 10_000_000, replace=False)
    else:
        sample_probs = flat_probs

    # Calculate percentiles to define search range
    p_start = np.percentile(sample_probs, Config.THRESHOLD_SEARCH_START)
    p_end = np.percentile(sample_probs, Config.THRESHOLD_SEARCH_END)

    print(
        f"Search Range (Percentiles {Config.THRESHOLD_SEARCH_START}-{Config.THRESHOLD_SEARCH_END}): {p_start} to {p_end}"
    )

    thresholds = np.linspace(p_start, p_end, Config.THRESHOLD_SEARCH_STEPS)

    best_threshold = 0.5
    best_score = -1.0

    # Iterate through thresholds to find the maximum F1 score
    for thresh in thresholds:
        # Binarize predictions
        y_pred = (y_probs > thresh).astype(int)

        # Calculate Mean F1-Score (Samples average)
        # zero_division=0 handles cases where a sample has no predicted tags and no true tags
        score = f1_score(y_true, y_pred, average="samples", zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    print(f"Best Threshold: {best_threshold} with Mean F1-Score: {best_score}")
    return best_threshold


def generate_submission(model, test_loader, test_ids, encoder, threshold):
    """
    Generates predictions for the test set using the optimized threshold,
    decodes the tags, and saves the submission file.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        test_ids (np.ndarray): Array of test question IDs.
        encoder (TagEncoder): The fitted TagEncoder for decoding tags.
        threshold (float): The decision threshold.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    model.to(device)
    model.eval()

    print(f"Generating submission with threshold {threshold}...")

    all_probs = []

    # 1. Inference
    with Timer("Test Inference"):
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device, non_blocking=True)

                # Forward pass
                logits = model(inputs)
                probs = torch.sigmoid(logits)

                # Move to CPU and collect
                all_probs.append(probs.cpu().numpy())

    if all_probs:
        all_probs = np.vstack(all_probs)
    else:
        all_probs = np.zeros((0, Config.OUTPUT_DIM))

    # 2. Thresholding and Decoding
    print("Applying threshold and decoding tags...")
    with Timer("Decoding"):
        # Create binary matrix
        binary_preds = (all_probs > threshold).astype(int)

        # Inverse transform to get list of tuples/lists of tags
        predicted_tags_tuples = encoder.inverse_transform(binary_preds)

        # Join tags with space to match submission format
        predicted_tags_strings = [" ".join(tags) for tags in predicted_tags_tuples]

    # 3. Create Submission DataFrame
    submission_df = pd.DataFrame({"Id": test_ids, "Tags": predicted_tags_strings})

    # 4. Save to CSV
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")

    return submission_df
