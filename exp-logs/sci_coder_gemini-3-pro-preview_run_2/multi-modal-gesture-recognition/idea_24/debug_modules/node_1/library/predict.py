import os
import torch
import numpy as np
import scipy.signal
import pandas as pd
from torch.utils.data import DataLoader

from library.config import (
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    BATCH_SIZE,
    MEDIAN_FILTER_KERNEL,
    SEED,
    NUM_CLASSES,
)
from library.utils import set_seed
from library.dataset import GestureDataset, collate_fn
from library.model import SBG_CRCN


def apply_median_filter(predictions, kernel_size):
    """
    Applies a median filter to the prediction sequence with nearest-neighbor padding.

    Args:
        predictions (np.ndarray): 1D array of frame labels.
        kernel_size (int): Size of the median filter window. Must be odd.

    Returns:
        np.ndarray: Smoothed predictions.
    """
    if kernel_size <= 1:
        return predictions

    # Ensure kernel size is odd
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    pad_width = k // 2

    if len(predictions) == 0:
        return predictions

    # Nearest-neighbor padding: pad with the first and last values
    # mode='edge' in np.pad corresponds to nearest neighbor padding
    padded_preds = np.pad(predictions, pad_width, mode="edge")

    # Apply median filter
    smoothed = scipy.signal.medfilt(padded_preds, kernel_size=k)

    # Remove padding (medfilt returns same size as input, so we slice the middle)
    # Actually scipy.signal.medfilt returns an array of the same size as input.
    # Since we padded the input manually, the output will be size N + 2*pad.
    # We need to slice out the center.
    start = pad_width
    end = start + len(predictions)
    return smoothed[start:end]


def decode_predictions_robust(class_probs, mask, kernel_size=MEDIAN_FILTER_KERNEL):
    """
    Decodes frame-level probabilities into a sequence of gesture labels.

    Args:
        class_probs (torch.Tensor): (B, T, C) Softmax probabilities.
        mask (torch.Tensor): (B, T) Boolean mask.
        kernel_size (int): Kernel size for median filtering.

    Returns:
        list of lists: Predicted gesture sequences for the batch.
    """
    predictions = []

    # Convert to numpy for processing
    class_probs_np = class_probs.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    B, T, C = class_probs_np.shape

    for b in range(B):
        # Get valid length
        valid_len = int(np.sum(mask_np[b]))
        if valid_len == 0:
            predictions.append([])
            continue

        # Get frame labels: Argmax
        probs_b = class_probs_np[b, :valid_len, :]
        frame_labels = np.argmax(probs_b, axis=1)

        # Apply Median Filter to smooth
        filtered_labels = apply_median_filter(frame_labels, kernel_size)

        # Collapse repeats and remove background (0)
        sequence = []
        last_label = -1

        for label in filtered_labels:
            if label != last_label:
                if label != 0:  # 0 is background
                    sequence.append(int(label))
                last_label = label

        predictions.append(sequence)

    return predictions


def generate_submission(
    checkpoint_path=None,
    output_file="submission.csv",
    batch_size=BATCH_SIZE,
    load_cached_data=True,
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        checkpoint_path (str): Path to the model checkpoint. If None, uses best_model.pth.
        output_file (str): Name of the output CSV file.
        batch_size (int): Batch size for inference.
        load_cached_data (bool): Whether to use cached pre-processed data.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Paths
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    output_path = os.path.join(SUBMISSION_DIR, output_file)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # 2. Load Data
    print("Loading Test Data...")
    test_dataset = GestureDataset(
        split="test", augment=False, load_cached_data=load_cached_data
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Load Model
    print(f"Loading Model from {checkpoint_path}...")
    model = SBG_CRCN().to(device)

    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return

    model.eval()

    # 4. Inference Loop
    print("Starting Inference...")
    results = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            sample_ids = batch["sample_ids"]

            # Forward Pass
            outputs = model(features, mask)

            # Use Stage 3 output (most refined)
            stage3_probs = outputs["stage3"]["class_probs"]

            # Decode
            batch_preds = decode_predictions_robust(stage3_probs, mask)

            # Store results
            for sid, pred_seq in zip(sample_ids, batch_preds):
                # Format: SessionID,label1,label2,...
                pred_str = ",".join(map(str, pred_seq))
                results.append((sid, pred_str))

    # 5. Write Submission
    print(f"Writing submission to {output_path}...")

    # Sort by SessionID to be tidy (optional but good practice)
    results.sort(key=lambda x: x[0])

    try:
        with open(output_path, "w") as f:
            for sid, pred_str in results:
                if pred_str:
                    line = f"{sid},{pred_str}\n"
                else:
                    # If no gestures detected, just SessionID
                    line = f"{sid}\n"
                f.write(line)
        print("Submission generated successfully.")

    except Exception as e:
        print(f"Error writing submission file: {e}")
