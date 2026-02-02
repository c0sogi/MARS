import os
import numpy as np
import pandas as pd
import torch
from library.config import DEVICE, SUBMISSION_FILE, TEST_METADATA
from library.utils import rle_encoding, fbeta_score


def optimize_threshold(model, val_loader, device=DEVICE):
    """
    Finds the probability threshold that maximizes the F0.5 score on the validation set.

    Args:
        model (nn.Module): The trained model.
        val_loader (DataLoader): DataLoader for the validation set.
        device (str): Device to run inference on.

    Returns:
        float: The optimal threshold value.
    """
    model.eval()
    all_preds = []
    all_targets = []

    # Collect predictions and targets
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            logits = model(inputs)
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    if not all_preds:
        print("No validation data available. Defaulting threshold to 0.5.")
        return 0.5

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    best_threshold = 0.5
    best_score = -1.0

    # Search range [0.1, 0.9] with step 0.05
    thresholds = np.arange(0.1, 0.95, 0.05)

    for thr in thresholds:
        score = fbeta_score(all_preds, all_targets, beta=0.5, threshold=thr)
        if score > best_score:
            best_score = score
            best_threshold = thr

    print(f"Optimized Threshold: {best_threshold}")
    print(f"Best Val F0.5: {best_score}")

    return best_threshold


def predict_fragment(model, test_loader, device=DEVICE):
    """
    Performs sliding-window inference on test volumes and stitches the resulting patches
    into full-size probability maps.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.

    Returns:
        dict: A dictionary mapping fragment_ids to their reconstructed probability map (numpy array).
    """
    model.eval()

    # 1. Determine canvas sizes for each test fragment from metadata
    if not os.path.exists(TEST_METADATA):
        raise FileNotFoundError(f"Test metadata not found at {TEST_METADATA}")

    df_test = pd.read_csv(TEST_METADATA)
    fragment_ids = df_test["fragment_id"].unique()

    # Dictionary to store reconstructed probability maps
    fragment_maps = {}

    for fid in fragment_ids:
        fid_df = df_test[df_test["fragment_id"] == fid]
        # Calculate full dimensions: max coordinate + patch dimension
        # The metadata generation script ensures x, y are top-left coordinates.
        max_h = (fid_df["y"] + fid_df["h"]).max()
        max_w = (fid_df["x"] + fid_df["w"]).max()

        fragment_maps[fid] = np.zeros((max_h, max_w), dtype=np.float32)

    # 2. Inference Loop
    with torch.no_grad():
        for inputs, sample_ids in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.sigmoid(logits).cpu().numpy()

            # Iterate through the batch
            for i, sample_id in enumerate(sample_ids):
                # Parse sample_id: {fragment_id}_{y}_{x}
                # Using rsplit to handle potential underscores in fragment_id safely
                parts = sample_id.rsplit("_", 2)
                if len(parts) != 3:
                    continue

                fid, y_str, x_str = parts
                y, x = int(y_str), int(x_str)

                # Get prediction patch (Shape: H, W)
                pred_patch = preds[i, 0, :, :]
                h, w = pred_patch.shape

                if fid in fragment_maps:
                    # Stitch into canvas
                    # Boundary handling (ensure we don't exceed allocated canvas)
                    map_h, map_w = fragment_maps[fid].shape

                    h_end = min(y + h, map_h)
                    w_end = min(x + w, map_w)

                    h_len = h_end - y
                    w_len = w_end - x

                    if h_len > 0 and w_len > 0:
                        # Overwrite the region
                        fragment_maps[fid][y : y + h_len, x : x + w_len] = pred_patch[
                            :h_len, :w_len
                        ]

    return fragment_maps


def generate_submission_file(fragment_maps, threshold, submission_path=SUBMISSION_FILE):
    """
    Generates the submission.csv file by thresholding probability maps and applying RLE.

    Args:
        fragment_maps (dict): Dictionary of {fragment_id: probability_map}.
        threshold (float): Threshold to binarize predictions.
        submission_path (str): Path to save the CSV file.
    """
    submission_data = []

    # Process fragments in sorted order for consistent output
    for fid in sorted(fragment_maps.keys()):
        prob_map = fragment_maps[fid]

        # Binarize
        binary_mask = (prob_map > threshold).astype(np.uint8)

        # Run-Length Encoding
        rle = rle_encoding(binary_mask)

        submission_data.append({"Id": fid, "Predicted": rle})

    # Create DataFrame and save
    df_sub = pd.DataFrame(submission_data)

    # Ensure directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path}")
