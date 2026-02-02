import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from library import config, model, data_loader, utils

# Set fixed seeds
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)


def generate_submission(model_path=None, output_path=None, limit=None):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV.
        limit (int, optional): Limit the number of test samples for debugging.
    """
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        model_path = os.path.join(config.WORKING_DIR, "idea_34", "best_model.pth")

    if output_path is None:
        output_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # 2. Load Data
    # We instantiate the dataset manually to access metadata (sample_map, window_indices)
    test_ds = data_loader.GestureDataset(config.TEST_METADATA_PATH, "test", limit=limit)

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Load Model
    net = model.PG_HCKN().to(device)

    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model path {model_path} not found. Using random weights.")

    net.eval()

    # 4. Prepare Reconstruction Buffers
    # sample_map is a list of dicts: {'sample_id': str, 'num_frames': int, 'start_idx': int, ...}
    # We need to map sample_id to its length and its global start offset
    sample_lengths = {
        item["sample_id"]: item["num_frames"] for item in test_ds.sample_map
    }
    sample_offsets = {
        item["sample_id"]: item["start_idx"] for item in test_ds.sample_map
    }

    # Buffers to accumulate probabilities from overlapping windows
    seq_probs = {
        sid: np.zeros((length, config.NUM_CLASSES), dtype=np.float32)
        for sid, length in sample_lengths.items()
    }
    seq_counts = {
        sid: np.zeros((length, 1), dtype=np.float32)
        for sid, length in sample_lengths.items()
    }

    window_indices = test_ds.window_indices
    global_idx = 0

    # 5. Inference Loop
    with torch.no_grad():
        for features, _, _ in test_loader:
            features = features.to(device)

            # Forward pass
            outputs = net(features)
            # Use Stage 3 output for final prediction
            logits = outputs["stage3"]
            probs = F.softmax(logits, dim=2).cpu().numpy()

            batch_size = features.size(0)

            for i in range(batch_size):
                if global_idx >= len(window_indices):
                    break

                # Retrieve window metadata
                start_global, end_global, sid, needs_padding = window_indices[
                    global_idx
                ]

                # Get probabilities for this window
                window_prob = probs[i]  # (WindowSize, Classes)

                # Calculate relative indices within the specific sequence
                global_offset = sample_offsets[sid]

                if needs_padding:
                    # If padded, the valid data starts at 0 and goes up to actual_len
                    # (Logic: Dataset pads the END of the sequence to fit window)
                    actual_len = sample_lengths[sid]
                    valid_len = min(config.WINDOW_SIZE, actual_len)

                    # Slice the valid part of the prediction
                    window_prob = window_prob[:valid_len]

                    # Target slice in the reconstruction buffer
                    target_slice = slice(0, valid_len)
                else:
                    # Standard sliding window
                    rel_start = start_global - global_offset
                    rel_end = end_global - global_offset
                    target_slice = slice(rel_start, rel_end)

                # Accumulate
                if sid in seq_probs:
                    # Verify shapes to prevent indexing errors
                    buffer_slice_shape = seq_probs[sid][target_slice].shape
                    if buffer_slice_shape[0] == window_prob.shape[0]:
                        seq_probs[sid][target_slice] += window_prob
                        seq_counts[sid][target_slice] += 1.0

                global_idx += 1

    # 6. Decode and Format Submission
    results = []

    # Sort IDs for consistent output order
    sorted_sids = sorted(seq_probs.keys())

    for sid in sorted_sids:
        prob_sum = seq_probs[sid]
        count = seq_counts[sid]

        # Average probabilities (avoid division by zero)
        count[count == 0] = 1.0
        avg_probs = prob_sum / count

        # Frame-wise classification
        frame_preds = np.argmax(avg_probs, axis=1)

        # Run-Length Encoding with filtering
        # This handles background class removal and min duration filtering
        gestures = utils.run_length_encoding(
            frame_preds, min_duration=config.MIN_GESTURE_DURATION
        )

        # Format: SessionID,Label1,Label2,...
        if gestures:
            str_gestures = ",".join([str(g) for g in gestures])
            line = f"{sid},{str_gestures}"
        else:
            # If no gestures detected, just the ID (or ID with trailing comma if preferred,
            # but usually ID is sufficient or ID, for empty)
            # Based on example "Session00001,2,12,3", if empty likely just "Session00001"
            line = f"{sid}"

        results.append(line)

    # 7. Save to File
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Submission generated with {len(results)} samples.")
    print(f"Saved to: {output_path}")
