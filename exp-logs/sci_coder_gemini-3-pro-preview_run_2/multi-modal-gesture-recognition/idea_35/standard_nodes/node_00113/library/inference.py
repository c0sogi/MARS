import os
import torch
import pandas as pd
import numpy as np
import scipy.signal
from library.config import Config
from library.model import CRGCN
from library.data_loader import get_dataloaders
from library.utils import setup_logger


def decode_predictions(logits, lengths):
    """
    Decodes frame-wise logits into a list of gesture sequences.
    Applies Median Filtering to the discrete class labels, collapses
    consecutive repeats, and removes the background class (0).

    Args:
        logits (torch.Tensor): (B, C, T) tensor of model outputs.
        lengths (torch.Tensor): (B,) tensor of sequence lengths.

    Returns:
        list of list of int: Decoded gesture sequences for the batch.
    """
    # Get class indices: (B, T)
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    decoded_sequences = []

    for i in range(preds.shape[0]):
        length = lengths[i]
        raw_pred = preds[i, :length]

        # Determine window size for Median Filter (must be odd)
        window_size = Config.MEDIAN_WINDOW_SIZE
        if window_size % 2 == 0:
            window_size += 1

        # Apply Median Filter if sequence is long enough
        if len(raw_pred) < window_size:
            smoothed_pred = raw_pred
        else:
            smoothed_pred = scipy.signal.medfilt(raw_pred, kernel_size=window_size)

        # Collapse repeats and remove background
        sequence = []
        prev_label = -1

        for label in smoothed_pred:
            if label != prev_label:
                if label != 0:  # 0 is background
                    sequence.append(int(label))
                prev_label = label

        decoded_sequences.append(sequence)

    return decoded_sequences


def generate_submission(debug_size=None):
    """
    Runs inference on the test set using the best checkpoint and generates
    a submission CSV file.

    Args:
        debug_size (int, optional): Number of samples to process for debugging.
    """
    logger = setup_logger("inference")
    logger.info("Starting inference pipeline...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Data
    # We only need the test_loader. get_dataloaders returns (train, val, test).
    logger.info("Loading test data...")
    _, _, test_loader = get_dataloaders(debug_size=debug_size)

    # Retrieve Sample IDs from metadata to ensure correct alignment
    # test_loader.dataset is a GestureDataset which holds the metadata DataFrame
    test_metadata = test_loader.dataset.metadata
    sample_ids = test_metadata["sample_id"].tolist()

    logger.info(f"Loaded {len(sample_ids)} test samples.")

    # Initialize Model
    model = CRGCN().to(device)

    # Load Checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        logger.warning(
            f"Checkpoint not found at {checkpoint_path}. Using random initialization (DEBUG ONLY)."
        )

    model.eval()

    all_predictions = []

    # Inference Loop
    logger.info("Running prediction loop...")
    with torch.no_grad():
        for batch_idx, (features, _, _, mask, lengths) in enumerate(test_loader):
            features = features.to(device)
            mask = mask.to(device)

            # Forward pass
            outputs = model(features, mask)

            # Extract Stage 3 Classification Logits
            # outputs['stage3'] is a tuple (cls_logits, bnd_logits)
            stage3_cls_logits = outputs["stage3"][0]

            # Decode batch predictions
            batch_preds = decode_predictions(stage3_cls_logits, lengths)
            all_predictions.extend(batch_preds)

    # Validate Sample Count
    if len(all_predictions) != len(sample_ids):
        logger.error(
            f"Prediction count mismatch! Samples: {len(sample_ids)}, Predictions: {len(all_predictions)}"
        )
        # In case of mismatch (e.g. skipped corrupt files), we slice sample_ids to match
        # assuming the dataset loader skipped them sequentially.
        sample_ids = sample_ids[: len(all_predictions)]

    # Write Submission File
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    logger.info(f"Writing results to {submission_path}")

    with open(submission_path, "w") as f:
        for i, seq in enumerate(all_predictions):
            sid = sample_ids[i]
            # Format: SessionID,Label1,Label2,...
            # Example: Session00001,2,12,3
            labels_str = ",".join(map(str, seq))

            if labels_str:
                line = f"{sid},{labels_str}\n"
            else:
                # Handle empty predictions (no gestures detected)
                line = f"{sid}\n"

            f.write(line)

    logger.info("Submission generation complete.")
