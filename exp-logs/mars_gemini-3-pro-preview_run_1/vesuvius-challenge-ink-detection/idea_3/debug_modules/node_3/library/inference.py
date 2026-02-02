import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config, set_seed
from library.utils import rle_encode
from library.dataset import InkDataset
from library.engine import evaluate


def calibrate_threshold(model, val_loader, device):
    """
    Calculates the optimal threshold for F0.5 score using the validation set.

    Args:
        model (nn.Module): The trained model.
        val_loader (DataLoader): DataLoader for validation data.
        device (str): Compute device.

    Returns:
        float: The optimal threshold value.
    """
    print("Calibrating threshold on validation set...")
    # We use BCEWithLogitsLoss as the criterion, consistent with training
    # The weight doesn't affect the score calculation in evaluate, only the loss
    criterion = torch.nn.BCEWithLogitsLoss()

    # evaluate returns (avg_loss, best_score, best_threshold)
    val_loss, best_score, best_threshold = evaluate(
        model, val_loader, criterion, device
    )

    print(f"Calibration Complete. Val Loss: {val_loss:.6f}")
    print(f"Best F0.5 Score: {best_score:.6f} at Threshold: {best_threshold:.4f}")

    return best_threshold


def predict_fragment(model, fragment_id, metadata_df, device):
    """
    Performs inference on a single fragment by stitching patches.

    Args:
        model (nn.Module): The trained model.
        fragment_id (str): The ID of the fragment to predict.
        metadata_df (pd.DataFrame): DataFrame containing test metadata.
        device (str): Compute device.

    Returns:
        tuple: (probability_map, validity_mask)
            - probability_map (np.ndarray): Probability map of shape (H, W).
            - validity_mask (np.ndarray): Binary mask of valid pixels (H, W).
    """
    # Filter metadata for the specific fragment
    frag_meta = (
        metadata_df[metadata_df["fragment_id"] == fragment_id]
        .copy()
        .reset_index(drop=True)
    )

    if frag_meta.empty:
        raise ValueError(f"No metadata found for fragment {fragment_id}")

    # Load the fragment's validity mask to determine canvas dimensions
    # We assume the mask path is consistent across patches for the same fragment
    rel_mask_path = frag_meta.iloc[0]["mask_path"]
    full_mask_path = os.path.join(Config.INPUT_DIR, rel_mask_path)

    if not os.path.exists(full_mask_path):
        raise FileNotFoundError(f"Mask file not found: {full_mask_path}")

    # Load mask in grayscale
    validity_mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
    if validity_mask is None:
        raise ValueError(f"Failed to load mask image: {full_mask_path}")

    H, W = validity_mask.shape

    # Initialize probability map
    prob_map = np.zeros((H, W), dtype=np.float32)

    # Create dataset and loader for this fragment
    # Use batch_size=1 to handle potential variable patch sizes at edges
    dataset = InkDataset(frag_meta, mode="test", load_cached_data=True)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()  # Shape: (1, 1, h, w)

            # Extract coordinates
            x_start = batch["x"].item()
            y_start = batch["y"].item()
            w = batch["w"].item()
            h = batch["h"].item()

            # Get prediction for the patch
            # Remove batch and channel dims -> (h, w)
            patch_prob = probs[0, 0]

            # Place on canvas
            # Ensure dimensions match (they should by definition of the dataset logic)
            prob_map[y_start : y_start + h, x_start : x_start + w] = patch_prob[:h, :w]

    return prob_map, validity_mask


def generate_submission(model, device):
    """
    Generates the submission file for the competition.

    1. Calibrates threshold using validation set.
    2. Predicts on all test fragments.
    3. Applies threshold and RLE encoding.
    4. Saves to submission.csv.

    Args:
        model (nn.Module): The trained model.
        device (str): Compute device.
    """
    set_seed(Config.SEED)

    # --- 1. Calibration ---
    # Load validation metadata
    if os.path.exists(Config.VAL_METADATA):
        val_df = pd.read_csv(Config.VAL_METADATA)
        # Use batch_size=1 for safety against variable patch sizes
        val_dataset = InkDataset(val_df, mode="val", load_cached_data=True)
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
        )

        threshold = calibrate_threshold(model, val_loader, device)
    else:
        print("Validation metadata not found. Using default threshold 0.5.")
        threshold = 0.5

    # --- 2. Inference & Submission ---
    print(f"Generating submission with threshold: {threshold:.4f}")

    if not os.path.exists(Config.TEST_METADATA):
        print("Test metadata not found. Skipping inference.")
        return

    test_df = pd.read_csv(Config.TEST_METADATA)
    fragment_ids = sorted(test_df["fragment_id"].unique())

    submission_rows = []

    for fid in fragment_ids:
        print(f"Processing test fragment: {fid}")

        # Generate probability map
        prob_map, validity_mask = predict_fragment(model, fid, test_df, device)

        # Apply threshold to get binary mask
        binary_map = (prob_map > threshold).astype(np.uint8)

        # Mask out invalid regions (background)
        # validity_mask is 0 or 255 (or 1), normalize to 0/1
        mask_binary = (validity_mask > 0).astype(np.uint8)
        final_mask = binary_map * mask_binary

        # Run-Length Encode
        rle_str = rle_encode(final_mask)

        submission_rows.append({"Id": fid, "Predicted": rle_str})

    # Create DataFrame and save
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
