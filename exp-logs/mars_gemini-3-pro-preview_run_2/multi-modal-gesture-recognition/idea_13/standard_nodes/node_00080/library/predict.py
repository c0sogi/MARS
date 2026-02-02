import os
import torch
import numpy as np
import pandas as pd
import re
from library.config import Config
from library.utils import set_seed, get_device
from library.model import SBMD_CRCN
from library.data_loader import get_dataloaders
from library.train import decode_sequence


def post_process_sequence(frame_labels):
    """
    Applies post-processing to the raw frame-wise label sequence.
    This includes median filtering (smoothing), collapsing consecutive duplicates,
    and removing the background class (0).

    Args:
        frame_labels (np.ndarray): Sequence of raw frame labels.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    # We utilize the decode_sequence function from library.train to ensure
    # consistency with the validation logic used during training.
    return decode_sequence(frame_labels)


def generate_predictions(load_cached_data=True, limit=None):
    """
    Generates predictions for the test dataset using the best trained model.
    Saves the results to the submission file specified in Config.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        limit (int, optional): Limit the number of samples for debugging purposes.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Load Data
    # get_dataloaders returns (train, val, test). We only need test.
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Load Model
    print("Initializing model...")
    model = SBMD_CRCN().to(device)

    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}...")
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print(
            f"Warning: Best model not found at {Config.BEST_MODEL_PATH}. Using random weights (expect poor performance)."
        )

    model.eval()

    # 4. Inference
    results = []
    print("Running inference on test set...")

    with torch.no_grad():
        count = 0
        for batch in test_loader:
            if limit is not None and count >= limit:
                break

            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            sample_ids = batch["sample_ids"]

            # Forward pass
            outputs = model(features, mask)

            # Use Stage 3 Classification Head for final predictions
            logits = outputs["stage3_cls"]  # (B, T, C)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)  # (B, T)

            # Convert to CPU numpy
            preds_np = preds.cpu().numpy()
            mask_np = mask.cpu().numpy()

            # Process batch
            for i, sample_id in enumerate(sample_ids):
                # Get valid sequence length
                valid_len = int(mask_np[i].sum())

                # Slice valid frames
                pred_seq_raw = preds_np[i, :valid_len]

                # Post-process (Smooth -> Collapse -> Remove Background)
                final_gestures = post_process_sequence(pred_seq_raw)

                # Format output: Id,Sequence
                # Match randomPredictions.csv format: Id is integer, Sequence is space-separated
                try:
                    # Extract digits from the string (e.g. Sample00300 -> 300)
                    id_match = re.search(r"\d+", str(sample_id))
                    clean_id = int(id_match.group()) if id_match else 0
                except Exception:
                    clean_id = 0

                # Serialize sequence to space-separated string
                labels_str = " ".join(map(str, final_gestures))

                # Create CSV line: Id,Sequence
                line_str = f"{clean_id},{labels_str}"

                results.append((clean_id, line_str))
                count += 1

    # 5. Save Submission
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Sort results by Integer ID for tidiness
    results.sort(key=lambda x: x[0])

    with open(Config.SUBMISSION_PATH, "w") as f:
        f.write("Id,Sequence\n")
        for _, line in results:
            f.write(line + "\n")

    print(f"Predictions generated for {len(results)} sequences.")
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
