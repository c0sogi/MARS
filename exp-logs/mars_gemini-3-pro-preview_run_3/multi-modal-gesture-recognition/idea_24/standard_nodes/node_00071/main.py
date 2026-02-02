import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import random

# ==========================================
# 1. Configuration Override & Imports
# ==========================================
# We import config first to override settings for a fast baseline
from library import config

# Override for fast baseline execution
config.NUM_EPOCHS = 40  # Sufficient for convergence on small dataset
config.BATCH_SIZE = 32
config.DEBUG = False  # Ensure we use full data

from library import trainer, utils, model, dataset


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(config.SEED)

# ==========================================
# 2. Helper Functions
# ==========================================


def levenshtein_distance(seq1, seq2):
    """
    Computes Levenshtein distance between two sequences.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1
                )
    return matrix[size_x - 1, size_y - 1]


def predict_sequence(model_instance, features, device, window_size=64, stride=16):
    """
    Performs sliding window inference on a single sequence.
    Args:
        model_instance: Trained PyTorch model.
        features: (T, InputDim) numpy array.
        device: Torch device.
    Returns:
        predicted_labels: List of class IDs.
    """
    model_instance.eval()
    seq_len = features.shape[0]
    num_classes = config.NUM_CLASSES

    # Prepare probability buffer
    # We add padding to handle edge cases if sequence is short
    pad_len = 0
    if seq_len < window_size:
        pad_len = window_size - seq_len
        # Zero pad
        padding = np.zeros((pad_len, features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, padding], axis=0)
        seq_len = features.shape[0]

    # Buffer for probabilities: (SeqLen, NumClasses)
    probs_sum = np.zeros((seq_len, num_classes), dtype=np.float32)
    counts = np.zeros((seq_len, 1), dtype=np.float32)

    # Generate windows
    windows = []
    indices = []

    num_windows = (seq_len - window_size) // stride + 1
    # Ensure we cover the end
    starts = list(range(0, (num_windows * stride), stride))
    if starts[-1] + window_size < seq_len:
        starts.append(seq_len - window_size)

    # Batch processing for inference speed
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i : i + batch_size]
            batch_windows = []

            for s in batch_starts:
                window = features[s : s + window_size]
                batch_windows.append(window)

            # Stack and convert to tensor
            input_tensor = torch.from_numpy(np.array(batch_windows)).to(device)

            # Forward pass
            _, _, logits_3 = model_instance(input_tensor)
            batch_probs = torch.softmax(logits_3, dim=2).cpu().numpy()

            # Accumulate
            for j, s in enumerate(batch_starts):
                probs_sum[s : s + window_size] += batch_probs[j]
                counts[s : s + window_size] += 1.0

    # Average probabilities
    # Avoid division by zero (should not happen due to logic)
    counts[counts == 0] = 1.0
    avg_probs = probs_sum / counts

    # Remove padding if added
    if pad_len > 0:
        avg_probs = avg_probs[:-pad_len]

    # Decode
    frame_preds = np.argmax(avg_probs, axis=1)

    # RLE
    final_sequence = utils.rle_encode(frame_preds)

    return final_sequence


# ==========================================
# 3. Main Execution
# ==========================================


def main():
    print("Starting runfile.py execution...")

    # --------------------------------------
    # A. Train Model
    # --------------------------------------
    print("Initializing Trainer...")
    trainer_instance = trainer.Trainer()

    print("Starting Training...")
    trainer_instance.fit()

    # --------------------------------------
    # B. Validation & Metric Calculation
    # --------------------------------------
    print("Performing Validation Evaluation...")

    # Load full validation sequences using utils.load_data
    # This gives us the preprocessed (T, 193) features
    X_val, Y_val, ids_val = utils.load_data(mode="val", load_cached_data=True)

    total_distance = 0.0
    total_truth_gestures = 0

    # For failure analysis
    sample_errors = []
    sample_lengths = []

    device = trainer_instance.device
    model_instance = trainer_instance.model

    print(f"Evaluating on {len(X_val)} validation sequences...")

    for i, (feats, label_seq) in enumerate(zip(X_val, Y_val)):
        # Ground Truth Sequence
        gt_sequence = utils.rle_encode(label_seq)
        total_truth_gestures += len(gt_sequence)

        # Prediction
        pred_sequence = predict_sequence(model_instance, feats, device)

        # Metric
        dist = levenshtein_distance(gt_sequence, pred_sequence)
        total_distance += dist

        # Store for analysis
        sample_errors.append(dist)
        sample_lengths.append(feats.shape[0])

    # Compute Final Metric
    # Avoid division by zero
    if total_truth_gestures == 0:
        final_metric = 0.0
    else:
        final_metric = total_distance / total_truth_gestures

    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------
    # C. Failure Analysis
    # --------------------------------------
    print("Performing Failure Analysis...")
    if len(sample_errors) > 0:
        correlation = np.corrcoef(sample_lengths, sample_errors)[0, 1]
        print(f"Correlation between Sequence Length and Error: {correlation:.4f}")

        # Basic stats
        avg_error = np.mean(sample_errors)
        print(f"Average Levenshtein Error per Sequence: {avg_error:.4f}")

    # --------------------------------------
    # D. Submission Generation
    # --------------------------------------
    THRESHOLD = 0.2251

    if final_metric < THRESHOLD:
        print(
            f"Validation metric ({final_metric:.4f}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        X_test, _, ids_test = utils.load_data(mode="test", load_cached_data=True)

        submission_lines = []

        print(f"Predicting on {len(X_test)} test sequences...")
        for i, (feats, sample_id) in enumerate(zip(X_test, ids_test)):
            pred_sequence = predict_sequence(model_instance, feats, device)

            # Format: SessionID,Label1,Label2...
            # Convert ints to strings
            str_preds = [str(p) for p in pred_sequence]
            line = f"{sample_id}," + ",".join(str_preds)
            submission_lines.append(line)

        # Write to file
        sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        with open(sub_path, "w") as f:
            for line in submission_lines:
                f.write(line + "\n")

        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"Validation metric ({final_metric:.4f}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
