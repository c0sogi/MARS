import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from library.config import (
    TEST_METADATA_PATH,
    INPUT_DIR,
    WORKING_DIR,
    WINDOW_SIZE,
    STRIDE,
    NUM_CLASSES,
    NUM_JOINTS,
    JOINTS_DIM,
    N_MFCC,
    MIN_GESTURE_DURATION,
    SEED,
)
from library.data_utils import load_robust_mat, compute_kinematics, extract_audio_mfcc
from library.model import KC_IRN

# Ensure deterministic behavior
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def calculate_levenshtein(seq1, seq2):
    """
    Calculates the Levenshtein distance between two sequences.
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
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1], matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1
                )
    return matrix[size_x - 1, size_y - 1]


def post_process_predictions(frame_indices):
    """
    Converts frame-wise class indices to a list of gesture IDs.
    Applies Run-Length Encoding and filters out short segments.

    Args:
        frame_indices (np.ndarray): Array of class indices (T,).

    Returns:
        list: Ordered list of recognized gesture IDs (int).
    """
    predictions = []
    if len(frame_indices) == 0:
        return predictions

    # Run-Length Encoding logic
    current_label = frame_indices[0]
    current_len = 1

    for i in range(1, len(frame_indices)):
        label = frame_indices[i]
        if label == current_label:
            current_len += 1
        else:
            # End of a segment
            # Check if it's a valid gesture (not background 0) and meets duration criteria
            if current_label != 0 and current_len >= MIN_GESTURE_DURATION:
                predictions.append(int(current_label))

            current_label = label
            current_len = 1

    # Handle the final segment
    if current_label != 0 and current_len >= MIN_GESTURE_DURATION:
        predictions.append(int(current_label))

    return predictions


def predict_sliding_window(model, features, device):
    """
    Performs sliding window inference on a single sequence.
    Aggregates probabilities from overlapping windows.

    Args:
        model (nn.Module): Trained model.
        features (np.ndarray): Input features (T, Input_Dim).
        device (torch.device): Computation device.

    Returns:
        np.ndarray: Frame-wise class probabilities (T, Num_Classes).
    """
    model.eval()
    seq_len, input_dim = features.shape

    # Prepare global buffers for aggregation
    # We use float32 for accumulation
    global_probs = np.zeros((seq_len, NUM_CLASSES), dtype=np.float32)
    counts = np.zeros((seq_len, 1), dtype=np.float32)

    # Handle sequences shorter than window size
    if seq_len < WINDOW_SIZE:
        pad_len = WINDOW_SIZE - seq_len
        # Repeat last frame for features
        last_frame = features[-1:]
        feat_pad = np.repeat(last_frame, pad_len, axis=0)
        feat_window = np.concatenate([features, feat_pad], axis=0)

        # Prepare batch
        batch_tensor = (
            torch.tensor(feat_window, dtype=torch.float32).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            _, _, out_3 = model(batch_tensor)
            # out_3 is log_softmax, convert to probs
            probs = torch.exp(out_3).cpu().numpy()[0]  # (Window, Classes)

        # Add to global buffer (truncate padding)
        global_probs += probs[:seq_len]
        counts += 1.0

    else:
        # Sliding Window Generation
        windows = []
        indices = []  # Store (start, end) for each window

        current_start = 0
        while current_start + WINDOW_SIZE <= seq_len:
            end = current_start + WINDOW_SIZE
            windows.append(features[current_start:end])
            indices.append((current_start, end))
            current_start += STRIDE

        # Handle the final window if not covered perfectly
        if current_start < seq_len:
            start = seq_len - WINDOW_SIZE
            end = seq_len
            # Avoid duplicating if we just added this exact window
            if not indices or indices[-1][0] != start:
                windows.append(features[start:end])
                indices.append((start, end))

        if not windows:
            return global_probs  # Should not happen given logic above

        # Batch processing
        # Depending on memory, we might want to process in smaller chunks.
        # Given the constraints (A100), processing all windows of one video is fine.
        batch_np = np.array(windows)
        batch_tensor = torch.tensor(batch_np, dtype=torch.float32).to(device)

        with torch.no_grad():
            _, _, out_3 = model(batch_tensor)
            batch_probs = (
                torch.exp(out_3).cpu().numpy()
            )  # (Num_Windows, Window_Size, Classes)

        # Aggregate
        for i, (start, end) in enumerate(indices):
            global_probs[start:end] += batch_probs[i]
            counts[start:end] += 1.0

    # Average
    # Avoid division by zero (though logic ensures counts >= 1)
    counts = np.maximum(counts, 1.0)
    avg_probs = global_probs / counts

    return avg_probs


def generate_submission(model_path, output_file, load_cached_data=True, limit=None):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model_path (str): Path to the trained model weights.
        output_file (str): Path to save the submission CSV.
        load_cached_data (bool): Whether to use cached features.
        limit (int): Optional limit on number of samples (for debugging).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")

    # 1. Load Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = KC_IRN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load Metadata
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found: {TEST_METADATA_PATH}")

    df = pd.read_csv(TEST_METADATA_PATH)
    if limit:
        df = df.head(limit)

    results = []

    print(f"Processing {len(df)} test sequences...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row["sample_id"]
        data_path = os.path.join(INPUT_DIR, row["data_path"])
        audio_path = os.path.join(INPUT_DIR, row["audio_path"])

        # --- Data Loading & Preprocessing ---
        # 1. Load Skeleton
        skeleton = load_robust_mat(data_path, load_cached_data=load_cached_data)

        if skeleton is None:
            # Fallback for missing data: predict empty
            results.append(f"{sample_id},")
            continue

        num_frames = skeleton.shape[0]
        if num_frames == 0:
            results.append(f"{sample_id},")
            continue

        # 2. Load Audio
        audio = extract_audio_mfcc(
            audio_path, num_frames, load_cached_data=load_cached_data
        )

        # 3. Compute Kinematics
        # (T, J, 3) -> (T, J, 9)
        kinematics = compute_kinematics(skeleton)

        # 4. Flatten Skeleton
        # (T, J, 9) -> (T, J*9)
        T, J, D = kinematics.shape
        kinematics_flat = kinematics.reshape(T, J * D)

        # 5. Fusion
        features = np.concatenate([kinematics_flat, audio], axis=-1)

        # --- Inference ---
        avg_probs = predict_sliding_window(model, features, device)

        # --- Post-Processing ---
        # Argmax to get class indices
        frame_preds = np.argmax(avg_probs, axis=1)

        # RLE and Filtering
        predicted_gestures = post_process_predictions(frame_preds)

        # Format string
        if predicted_gestures:
            pred_str = ",".join(map(str, predicted_gestures))
            line = f"{sample_id},{pred_str}"
        else:
            line = f"{sample_id},"

        results.append(line)

    # 3. Save Submission
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Submission saved to {output_file}")
