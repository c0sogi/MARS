import os
import torch
import torch.nn.functional as F
import numpy as np
from library import config, utils, model, data_loader


def sliding_window_inference(model, features, window_size, stride, device):
    """
    Performs inference on a variable length sequence using a sliding window approach.
    Accumulates probabilities from overlapping windows and averages them.

    Args:
        model: The trained PyTorch model.
        features: Tensor of shape (1, Time, InputDim).
        window_size: Int, size of the window.
        stride: Int, stride for the sliding window.
        device: Torch device.

    Returns:
        avg_probs: Tensor of shape (1, NumClasses, Time) containing averaged probabilities.
    """
    model.eval()
    seq_len = features.size(1)

    # Case 1: Sequence is shorter than window size
    if seq_len < window_size:
        pad_len = window_size - seq_len
        # Pad features at the end: (0, 0) for dim -1, (0, pad_len) for dim -2
        features_padded = F.pad(features, (0, 0, 0, pad_len))

        with torch.no_grad():
            outputs = model(features_padded.to(device))
            # outputs is [stage1, stage2, stage3]. We use Stage 3.
            log_probs = outputs[2]  # (1, Classes, Time)
            probs = torch.exp(log_probs)

        # Crop back to original length
        probs = probs[:, :, :seq_len]
        return probs

    # Case 2: Sequence is longer than window size
    # Initialize accumulators on device
    sum_probs = torch.zeros(1, config.NUM_CLASSES, seq_len, device=device)
    count_map = torch.zeros(1, 1, seq_len, device=device)

    current_t = 0
    while current_t < seq_len:
        start = current_t
        end = start + window_size

        # Handle the last window
        if end > seq_len:
            # Shift start back so the window ends exactly at seq_len
            start = max(0, seq_len - window_size)
            end = seq_len

        # Extract window
        window_feat = features[:, start:end, :]

        with torch.no_grad():
            outputs = model(window_feat.to(device))
            # Use Stage 3 output
            log_probs = outputs[2]
            probs = torch.exp(log_probs)

        # Accumulate
        sum_probs[:, :, start:end] += probs
        count_map[:, :, start:end] += 1.0

        # If we just processed the end of the sequence, break
        if end == seq_len:
            break

        current_t += stride

    # Average probabilities
    # Avoid division by zero (though count_map should be >= 1 everywhere)
    avg_probs = sum_probs / torch.clamp(count_map, min=1.0)

    return avg_probs


def predict_test_set(load_cached_data=True):
    """
    Main function to generate predictions for the test set.
    Loads model, runs inference, decodes predictions, and saves to CSV.
    """
    # 1. Setup
    device = config.DEVICE
    utils.set_seed(config.SEED)

    print(f"Running inference on device: {device}")

    # 2. Load Model
    net = model.KC_IRN().to(device)
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Loading model weights from {config.MODEL_SAVE_PATH}")
        net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    else:
        print(f"Error: Model file not found at {config.MODEL_SAVE_PATH}")
        return

    net.eval()

    # 3. Load Test Data
    # get_dataloaders returns (train, val, test). We only need test.
    print("Loading test data...")
    _, _, test_loader = data_loader.get_dataloaders(load_cached=load_cached_data)

    submission_lines = []
    print(f"Processing {len(test_loader)} test sequences...")

    # 4. Inference Loop
    for i, (features, _, sample_ids) in enumerate(test_loader):
        # features: (1, Time, InputDim)
        # sample_ids: tuple of size 1
        sample_id = sample_ids[0]

        # Run sliding window inference
        probs = sliding_window_inference(
            net,
            features,
            window_size=config.WINDOW_SIZE,
            stride=config.STRIDE,
            device=device,
        )

        # 5. Decode Predictions
        # Get class indices: (Time,)
        pred_labels = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()

        # Decode to ordered list of gesture IDs
        # Filter out background (0) and very short segments (<5 frames)
        sequence = utils.decode_predictions_to_sequence(
            pred_labels, background_id=config.BACKGROUND_CLASS_ID, min_len=5
        )

        # 6. Format Output
        # Format: Id,Sequence
        labels_str = " ".join(map(str, sequence))

        # Sanitize ID (Cite debug_lesson_5)
        clean_id = int("".join(filter(str.isdigit, str(sample_id))))

        line = f"{clean_id},{labels_str}"
        submission_lines.append(line)

    # 7. Save Submission
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    with open(config.SUBMISSION_PATH, "w") as f:
        f.write("Id,Sequence\n")
        for line in submission_lines:
            f.write(line + "\n")

    print(f"Submission saved to {config.SUBMISSION_PATH}")
