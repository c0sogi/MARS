import os
import torch
import numpy as np
from scipy.ndimage import median_filter
from library.config import DEVICE, CACHE_DIR, SUBMISSION_FILE
from library.model import DSG_CRCN
from library.data import get_loaders
from library.utils import format_submission, set_seed


def post_process_labels(pred_seq, kernel_size=15):
    """
    Applies a Median Filter with Nearest-Neighbor padding to the discrete predictions.

    Args:
        pred_seq (np.ndarray): 1D array of predicted labels.
        kernel_size (int): Size of the median filter window.

    Returns:
        np.ndarray: Filtered label sequence.
    """
    # mode='nearest' corresponds to Nearest-Neighbor Padding
    return median_filter(pred_seq, size=kernel_size, mode="nearest")


def decode_predictions(filtered_seq):
    """
    Collapses repeated labels and removes background classes.

    Args:
        filtered_seq (np.ndarray): 1D array of filtered labels.

    Returns:
        list: Ordered list of gesture IDs (integers).
    """
    final_seq = []
    prev_label = -1

    for label in filtered_seq:
        if label != prev_label:
            if label != 0:  # Remove background (0)
                final_seq.append(int(label))
            prev_label = label

    return final_seq


def predict_all(model, test_loader, device=DEVICE, output_path=SUBMISSION_FILE):
    """
    Runs the trained model on test data, applies post-processing, and saves the submission.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.
        output_path (str): Path to save the CSV submission.
    """
    model.eval()
    all_sample_ids = []
    all_predictions = []

    print("Generating predictions...")

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"]
            sample_ids = batch["sample_ids"]

            # Forward pass
            outputs = model(features, mask)

            # Use final stage classification output
            final_cls = outputs["final_cls"]  # (B, T, C)

            # Get raw class indices
            batch_preds = torch.argmax(final_cls, dim=2).cpu().numpy()  # (B, T)

            for i, raw_seq in enumerate(batch_preds):
                length = lengths[i]
                # Truncate to valid length to avoid processing padding
                valid_seq = raw_seq[:length]

                # --- Post-Processing ---
                # 1. Median Filter (Label-Space Smoothing)
                filtered_seq = post_process_labels(valid_seq)

                # 2. Decoding
                final_seq = decode_predictions(filtered_seq)

                all_sample_ids.append(sample_ids[i])
                all_predictions.append(final_seq)

    # Save to CSV
    format_submission(all_sample_ids, all_predictions, output_path)


def run_inference(checkpoint_path=None, load_cached_data=True):
    """
    Main entry point for the inference pipeline.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to best_model.pth in CACHE_DIR.
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    set_seed()

    # Initialize Datasets and Loaders
    # We only need the test loader here, but get_loaders returns all three
    print("Initializing data loaders...")
    _, _, test_loader = get_loaders(load_cached_data=load_cached_data)

    # Initialize Model
    model = DSG_CRCN().to(DEVICE)

    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CACHE_DIR, "best_model.pth")

    # Load Weights
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random initialization (predictions will be random)."
        )

    # Run Prediction
    predict_all(model, test_loader, device=DEVICE, output_path=SUBMISSION_FILE)
