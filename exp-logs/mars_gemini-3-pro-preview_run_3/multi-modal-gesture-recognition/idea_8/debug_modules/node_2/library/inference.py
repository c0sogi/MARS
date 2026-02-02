import sys
import os
import torch
import numpy as np
from collections import defaultdict

# Add library path to access provided modules
sys.path.append(os.path.abspath("./library"))
from config import Config
from project_utils import set_seed, run_length_encoding, save_submission
from model import BAKC_IRN
from data_loader import get_dataloaders


def predict_sequence(model, test_loader, device):
    """
    Runs inference on the test loader using the provided model.
    Aggregates sliding window predictions to form full sequence predictions.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Computation device.

    Returns:
        dict: A dictionary mapping sample_id to a list of predicted gesture IDs.
    """
    model.eval()
    sample_predictions = defaultdict(list)

    # 1. Sliding Window Inference
    with torch.no_grad():
        for features, _, _, sample_ids, starts in test_loader:
            features = features.to(device)

            # Forward pass through the network
            # outputs is a list of dicts from each stage
            outputs = model(features)

            # We use the output of the final refinement stage
            final_output = outputs[-1]
            cls_logits = final_output["cls"]  # Shape: (Batch, NumClasses, Time)

            # Move to CPU for aggregation
            cls_logits = cls_logits.cpu().numpy()
            starts = starts.numpy()

            batch_size = cls_logits.shape[0]

            for i in range(batch_size):
                sid = sample_ids[i]
                start_frame = starts[i]

                # Transpose to (Time, NumClasses) for easier indexing
                logits = cls_logits[i].transpose(1, 0)

                sample_predictions[sid].append((start_frame, logits))

    # 2. Sequence Reconstruction and Decoding
    final_sequences = {}

    for sid, fragments in sample_predictions.items():
        # Determine the total length of the sequence
        max_len = 0
        for start, logits in fragments:
            end = start + logits.shape[0]
            if end > max_len:
                max_len = end

        # Initialize array to accumulate logits
        # We sum logits from overlapping windows
        full_logits = np.zeros((max_len, Config.NUM_CLASSES), dtype=np.float32)

        for start, logits in fragments:
            length = logits.shape[0]
            full_logits[start : start + length] += logits

        # Frame-wise prediction via Argmax
        frame_preds = np.argmax(full_logits, axis=1)

        # Run Length Encoding (Collapsing duplicates and removing background)
        # No median filtering is applied, relying on the model's smoothness
        sequence = run_length_encoding(frame_preds)
        final_sequences[sid] = sequence

    return final_sequences


def run_inference():
    """
    Main entry point for generating the submission.
    Loads model, runs inference, and saves the CSV.
    """
    # Setup
    Config.setup_directories()
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running inference on device: {device}")

    # Load Data
    # We only need the test loader
    _, _, test_loader = get_dataloaders()
    print(f"Test loader initialized with {len(test_loader)} batches.")

    # Initialize Model
    model = BAKC_IRN().to(device)

    # Load Weights
    checkpoint_path = Config.BEST_MODEL_PATH
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}.")
        print(
            "Using initialized weights (random). This is expected only during dry-run testing without training."
        )

    # Generate Predictions
    print("Generating predictions...")
    predictions = predict_sequence(model, test_loader, device)

    # Save Submission
    output_path = Config.SUBMISSION_PATH
    print(f"Saving submission to {output_path}...")
    save_submission(predictions, output_path)
    print("Inference complete.")
