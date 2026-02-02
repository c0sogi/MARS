import os
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from library.config import Config
from library.model import GMG_CRGN
from library.data_loader import get_dataloaders
from library.utils import load_checkpoint


def predict_all(model, loader, device):
    """
    Runs inference on the entire dataset provided by the loader.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        list: A list of tuples (session_id, raw_predictions_numpy_array).
              raw_predictions_numpy_array has shape (T,) containing class indices.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch_idx, (features, _, lengths, mask, ids) in enumerate(loader):
            features = features.to(device)
            mask = mask.to(device)

            # Forward pass
            # outputs is a list of [Stage1, Stage2, ...]
            # We use the final stage output
            outputs = model(features, mask)
            final_stage_out = outputs[-1]  # (B, T, C+1)

            # Extract classification logits (ignore boundary channel at the end)
            cls_logits = final_stage_out[:, :, : Config.NUM_CLASSES]  # (B, T, 21)

            # Get class indices (Argmax)
            # We do this before moving to CPU to save bandwidth
            preds = torch.argmax(cls_logits, dim=2)  # (B, T)

            # Move to CPU
            preds_np = preds.cpu().numpy()
            lengths_np = lengths.cpu().numpy()

            # Unpack batch
            for i, seq_id in enumerate(ids):
                valid_len = lengths_np[i]
                # Slice to valid length to remove padding
                seq_preds = preds_np[i, :valid_len]
                results.append((seq_id, seq_preds))

    return results


def post_process_sequence(seq_preds, kernel_size=7):
    """
    Applies Label-Space Median Filtering and decodes the sequence.

    Args:
        seq_preds (np.array): Array of class indices (T,).
        kernel_size (int): Size of the median filter window.

    Returns:
        list: Ordered list of recognized gesture IDs (excluding background).
    """
    if len(seq_preds) == 0:
        return []

    # 1. Label-Space Smoothing
    # Apply Median Filter with Nearest-Neighbor Padding (mode='nearest')
    # This prevents edge artifacts and protects start/end gestures
    smoothed_preds = median_filter(seq_preds, size=kernel_size, mode="nearest")

    # 2. Decoding
    # Collapse consecutive repeats and remove background (Class 0)
    collapsed_gestures = []
    prev_label = -1

    for label in smoothed_preds:
        if label != prev_label:
            if label != 0:  # 0 is background
                collapsed_gestures.append(int(label))
            prev_label = label

    return collapsed_gestures


def generate_submission(
    checkpoint_name="best_model.pth",
    output_file="submission.csv",
    device=None,
    debug=False,
):
    """
    Main function to generate the submission file.

    Args:
        checkpoint_name (str): Name of the checkpoint file in the checkpoint directory.
        output_file (str): Name of the output CSV file.
        device (torch.device, optional): Device to run on.
        debug (bool): If True, runs on a subset of data.
    """
    if device is None:
        device = torch.device(Config.DEVICE)

    print(f"Running inference on device: {device}")

    # 1. Load Data
    # We only need the test loader
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=debug,
        load_cached_data=True,
    )

    # 2. Load Model
    print("Loading model...")
    model = GMG_CRGN().to(device)

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    load_checkpoint(model, checkpoint_path)

    # 3. Inference
    print("Predicting...")
    raw_results = predict_all(model, test_loader, device)

    # 4. Post-processing and Formatting
    print("Post-processing and generating submission...")
    submission_lines = []

    for seq_id, seq_preds in raw_results:
        # Apply median filter and decoding
        gestures = post_process_sequence(
            seq_preds, kernel_size=9
        )  # Using kernel size 9 for robustness

        # Format: SessionID,Label1,Label2,...
        gesture_str = ",".join(map(str, gestures))
        line = f"{seq_id},{gesture_str}"
        submission_lines.append(line)

    # 5. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(Config.SUBMISSION_DIR, output_file)

    with open(out_path, "w") as f:
        for line in submission_lines:
            f.write(line + "\n")

    print(f"Submission saved successfully to {out_path}")
    print(f"Total sequences processed: {len(submission_lines)}")
