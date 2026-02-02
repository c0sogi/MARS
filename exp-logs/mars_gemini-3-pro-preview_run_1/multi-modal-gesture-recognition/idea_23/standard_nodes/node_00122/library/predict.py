import os
import torch
import csv
import numpy as np
from library.config import Config
from library.utils import (
    set_seed,
    median_filter_predictions,
    decode_predictions_to_gestures,
)
from library.data_loader import get_dataloaders
from library.model import DW_AIIN


def generate_predictions(model, loader, device):
    """
    Runs inference on the test loader using the provided model.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Test data loader.
        device (torch.device): Computation device.

    Returns:
        dict: A dictionary mapping sample_id (str) to a list of gesture IDs (int).
    """
    model.eval()
    results = {}

    print("Starting inference on test set...")

    with torch.no_grad():
        for batch in loader:
            # Move data to device
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            lengths = batch["lengths"].to(device)
            sample_ids = batch["sample_ids"]

            # Forward pass
            # Output: (B, T, NumClasses)
            logits = model(skeleton, audio, lengths)

            # Get predictions
            probs = torch.softmax(logits, dim=2)
            preds = torch.argmax(probs, dim=2).cpu().numpy()

            # Process each sample in the batch
            for i, sample_id in enumerate(sample_ids):
                length = lengths[i].item()

                # Extract valid frames (ignore padding)
                valid_pred = preds[i, :length]

                # 1. Apply Median Filter
                smoothed_pred = median_filter_predictions(
                    valid_pred, window_size=Config.MEDIAN_FILTER_WINDOW
                )

                # 2. Decode to Gesture List (RLE + Filtering)
                gesture_sequence = decode_predictions_to_gestures(
                    smoothed_pred,
                    background_label=Config.BACKGROUND_LABEL,
                    min_length=Config.MIN_GESTURE_LENGTH,
                )

                results[sample_id] = gesture_sequence

    return results


def save_submission(results, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        results (dict): Dictionary mapping sample_id to list of gesture IDs.
        output_path (str): Path to save the CSV.
    """
    print(f"Saving submission to {output_path}...")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        # The format is simple: SessionID,label1,label2,...
        # We don't use a CSV writer with header because the format is custom/simple
        # and variable length columns are tricky with standard CSV writers.

        # However, to be safe and standard, we can just write lines.
        for sample_id, gestures in results.items():
            # Convert list of ints to string
            gestures_str = ",".join(map(str, gestures))

            if gestures_str:
                line = f"{sample_id},{gestures_str}\n"
            else:
                line = f"{sample_id}\n"

            f.write(line)

    print("Submission saved successfully.")


def run_prediction(load_cached_data=True):
    """
    Main function to run the prediction pipeline.

    Args:
        load_cached_data (bool): Whether to use cached data in the data loader.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Load Data
    # get_dataloaders returns train, val, test. We only need test.
    # It handles stats computation/loading internally.
    _, _, test_loader = get_dataloaders()

    # 3. Load Model
    model = DW_AIIN().to(device)

    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.BEST_MODEL_PATH}. Please train the model first."
        )

    print(f"Loading model weights from {Config.BEST_MODEL_PATH}")
    state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    # 4. Generate Predictions
    predictions = generate_predictions(model, test_loader, device)

    # 5. Save Submission
    save_submission(predictions, Config.SUBMISSION_PATH)
