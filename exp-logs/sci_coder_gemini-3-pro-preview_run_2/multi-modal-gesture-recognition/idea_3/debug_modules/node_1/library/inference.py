import os
import torch
import numpy as np
from scipy.ndimage import median_filter
from library.config import (
    BEST_MODEL_PATH,
    SUBMISSION_PATH,
    MEDIAN_FILTER_KERNEL,
)
from library.model import HybridGestureNet
from library.data_loader import get_dataloaders
from library.utils import set_seed


def decode_sequence(frame_labels):
    """
    Decodes frame-wise labels into a sequence of gesture IDs.
    Logic: Collapse consecutive duplicates, then remove background (0).

    Args:
        frame_labels (list or np.array): Sequence of frame labels.

    Returns:
        list: Filtered list of gesture IDs.
    """
    if len(frame_labels) == 0:
        return []

    # Collapse duplicates
    collapsed = [frame_labels[0]]
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != frame_labels[i - 1]:
            collapsed.append(frame_labels[i])

    # Remove background (class 0)
    gesture_sequence = [x for x in collapsed if x != 0]
    return gesture_sequence


def generate_submission(device=None):
    """
    Generates the submission file for the test set using the best trained model.

    Args:
        device (str, optional): Device to run inference on ('cuda' or 'cpu').
                                If None, automatically detects available device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed()

    print(f"Starting inference on device: {device}")

    # 1. Load Data
    # get_dataloaders returns (train, val, test). We only need test.
    _, _, test_loader = get_dataloaders()

    # 2. Load Model
    model = HybridGestureNet()
    if os.path.exists(BEST_MODEL_PATH):
        try:
            state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded model checkpoint from {BEST_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Using initialized model.")
    else:
        print(
            f"Warning: Checkpoint not found at {BEST_MODEL_PATH}. Using initialized model (random weights)."
        )

    model.to(device)
    model.eval()

    submission_lines = []

    # 3. Inference Loop
    with torch.no_grad():
        for batch_idx, (x, y, lengths, ids) in enumerate(test_loader):
            x = x.to(device)

            # Forward Pass
            # model returns (logits_stage1, logits_stage2)
            # We use the output of Stage 2 (Refinement) for final predictions
            _, logits2 = model(x, lengths)

            # Get frame-wise class predictions
            # Shape: (Batch, Time)
            preds = torch.argmax(logits2, dim=2).cpu().numpy()

            # Process each sample in the batch
            for i in range(len(ids)):
                sample_id = ids[i]
                length = lengths[i]

                # Extract valid sequence (remove padding)
                raw_seq = preds[i, :length]

                # 4. Post-processing
                # Apply Median Filter to smooth predictions
                # mode='nearest' ensures nearest-neighbor padding at boundaries
                filtered_seq = median_filter(
                    raw_seq, size=MEDIAN_FILTER_KERNEL, mode="nearest"
                )

                # Decode sequence (collapse duplicates, remove background)
                decoded_gestures = decode_sequence(filtered_seq)

                # 5. Format Output
                # Format: SessionID,Label1,Label2,...
                if len(decoded_gestures) > 0:
                    gestures_str = ",".join(map(str, decoded_gestures))
                    line = f"{sample_id},{gestures_str}"
                else:
                    # If no gestures detected, output just the SessionID
                    line = f"{sample_id}"

                submission_lines.append(line)

    # 6. Save Submission
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    try:
        with open(SUBMISSION_PATH, "w") as f:
            for line in submission_lines:
                f.write(line + "\n")
        print(f"Submission file generated successfully: {SUBMISSION_PATH}")
        print(f"Total sequences processed: {len(submission_lines)}")
    except Exception as e:
        print(f"Error writing submission file: {e}")
