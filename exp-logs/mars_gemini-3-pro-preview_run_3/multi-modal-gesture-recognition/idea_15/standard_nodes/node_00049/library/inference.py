import os
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List

from library.config import Config
from library.model import RSKARN
from library.data_loader import get_test_loader
from library.utils import set_seed, rle_encode, save_submission


def generate_predictions(
    checkpoint_path: str = None,
    batch_size: int = Config.BATCH_SIZE,
    debug: bool = Config.DEBUG,
) -> None:
    """
    Runs inference on the test set using the RSK-ARN model.
    Performs sliding window aggregation and RLE decoding to generate the submission file.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to Config.WORKING_DIR/best_model.pth.
        batch_size (int): Batch size for inference.
        debug (bool): If True, runs on a subset of data.
    """
    # 1. Setup
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    # 2. Load Model
    print(f"Loading model from {checkpoint_path}...")
    model = RSKARN()
    model.to(device)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Load Data
    print("Initializing test data loader...")
    test_loader = get_test_loader(batch_size=batch_size, debug=debug)
    dataset = test_loader.dataset

    # Container for raw window predictions
    # Maps sample_id -> list of (start_frame, probability_tensor)
    raw_predictions: Dict[str, List[tuple]] = defaultdict(list)

    print("Starting sliding window inference...")
    with torch.no_grad():
        for data, _, indices in test_loader:
            data = data.to(device)

            # Forward pass
            # We use the probabilities from Stage 3 (final refinement stage)
            outputs = model(data)
            probs = (
                outputs["probs_3"].cpu().numpy()
            )  # Shape: (Batch, Window_Size, Num_Classes)

            # Map outputs back to their specific video sequence and temporal location
            for i, idx in enumerate(indices):
                # Retrieve metadata for this window using the dataset index
                # Note: indices from DataLoader are tensors, convert to int
                idx_int = idx.item()
                meta = dataset.window_metadata[idx_int]

                sample_id = meta["sample_id"]
                start_frame = meta["start_frame"]

                # Store the probability map for this window
                raw_predictions[sample_id].append((start_frame, probs[i]))

    # 4. Aggregation & Decoding
    print(f"Aggregating predictions for {len(raw_predictions)} sequences...")
    final_submission = {}

    for sample_id, windows in raw_predictions.items():
        # Determine the full length of the sequence
        # The end of the sequence is the maximum (start + window_size) across all windows
        max_len = 0
        for start, p in windows:
            end = start + p.shape[0]
            if end > max_len:
                max_len = end

        # Allocate buffers for accumulation
        # full_probs: Sum of probabilities for each frame
        # counts: Number of windows contributing to each frame (for averaging)
        full_probs = np.zeros((max_len, Config.NUM_CLASSES), dtype=np.float32)
        counts = np.zeros((max_len, 1), dtype=np.float32)

        # Accumulate probabilities
        for start, p in windows:
            length = p.shape[0]
            full_probs[start : start + length] += p
            counts[start : start + length] += 1.0

        # Compute average probabilities
        # Avoid division by zero (though counts should be >= 1 for covered frames)
        counts[counts == 0] = 1.0
        avg_probs = full_probs / counts

        # Decode: Argmax -> RLE
        # 1. Get the most likely class for each frame
        frame_preds = np.argmax(avg_probs, axis=1)

        # 2. Run-Length Encoding to get the list of gestures
        # This handles collapsing duplicates and removing the background class (0)
        gesture_list = rle_encode(frame_preds)

        final_submission[sample_id] = gesture_list

    # 5. Save Submission
    save_submission(final_submission)
    print("Inference completed successfully.")
