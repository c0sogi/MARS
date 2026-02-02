import os
import torch
import numpy as np
from library.config import Config
from library.utils import set_seed, run_length_encoding
from library.data_loader import get_dataloaders
from library.model import RHCKN
from library.train import aggregate_predictions


def generate_predictions(model, loader, device, output_path):
    """
    Performs sliding window inference, aggregates probabilities,
    decodes sequences using RLE with duration filtering, and saves to CSV.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): The test data loader.
        device (torch.device): Device to run inference on.
        output_path (str): Path to save the submission CSV.
    """
    print("Starting inference and submission generation...")

    # 1. Aggregate sliding window predictions
    # This function handles the sliding window logic (50% overlap) and temporal averaging
    aggregated = aggregate_predictions(loader, model, device)

    submission_lines = []

    # Sort by sample ID to ensure deterministic order
    # The aggregated dictionary keys are dataset indices, but values contain the 'id' string
    sorted_indices = sorted(aggregated.keys(), key=lambda k: aggregated[k]["id"])

    for s_idx in sorted_indices:
        data = aggregated[s_idx]
        sample_id = data["id"]
        probs = data["probs"]

        # 2. Decode
        # Argmax to get frame-wise labels from the averaged probabilities
        frame_preds = np.argmax(probs, axis=1)

        # Run-Length Encoding with Min Duration Filter
        # Filters out segments shorter than Config.MIN_GESTURE_DURATION (5 frames)
        # Removes background class (0)
        pred_seq = run_length_encoding(
            frame_preds,
            min_duration=Config.MIN_GESTURE_DURATION,
            background_class=Config.BACKGROUND_CLASS_ID,
        )

        # 3. Format Line
        # Format: SessionID,label1,label2,...
        if not pred_seq:
            # Handle case with no detected gestures
            labels_str = ""
        else:
            labels_str = ",".join(map(str, pred_seq))

        line = f"{sample_id},{labels_str}"
        submission_lines.append(line)

    # 4. Save to File
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for line in submission_lines:
            f.write(line + "\n")

    print(f"Submission saved to {output_path}")


def main():
    """
    Main execution function for prediction.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Using device: {device}")

    # 2. Load Data
    # We only need the test loader. load_cached_data=True ensures we use pre-processed data if available.
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Initialize Model
    model = RHCKN().to(device)

    # 4. Load Weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading model from {Config.MODEL_SAVE_PATH}")
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model file {Config.MODEL_SAVE_PATH} not found. Predictions will be random."
        )

    # 5. Generate Submission
    generate_predictions(model, test_loader, device, Config.SUBMISSION_PATH)
