import os
import torch
import numpy as np
import pandas as pd
import itertools
from library.config import Config
from library.utils import set_seed, compute_normalized_levenshtein
from library.model import HNGKN
from library.data_loader import load_and_process_split, compute_normalization_stats


def preprocess_sequence(audio_raw, skel_raw, stats):
    """
    Transforms raw audio and skeleton data into the normalized feature tensor
    expected by the model. Replicates logic from GestureDataset.

    Args:
        audio_raw (np.ndarray): (T, 13)
        skel_raw (np.ndarray): (T, 20, 3) in meters
        stats (dict): Normalization statistics

    Returns:
        torch.Tensor: (T, InputDim)
    """
    # Ensure numeric types for inputs coming from object-array caches
    audio_raw = audio_raw.astype(np.float32)
    skel_raw = skel_raw.astype(np.float32)

    # 1. Kinematic Feature Computation
    pos = skel_raw
    vel = np.gradient(pos, axis=0)
    acc = np.gradient(vel, axis=0)

    # 2. Hierarchical Normalization
    # Audio Z-Score
    audio_mean = stats["audio_mean"]
    audio_std = stats["audio_std"]
    audio_norm = (audio_raw - audio_mean) / (audio_std + 1e-6)

    # Skeleton Scaling
    skel_pos_std = stats["skel_pos_std"]
    pos_norm = pos / (skel_pos_std + 1e-6)
    vel_norm = vel / (skel_pos_std + 1e-6)
    acc_norm = acc / (skel_pos_std + 1e-6)

    # Flatten skeleton features: (T, 20*3)
    T = pos.shape[0]
    pos_flat = pos_norm.reshape(T, -1)
    vel_flat = vel_norm.reshape(T, -1)
    acc_flat = acc_norm.reshape(T, -1)

    # Concatenate Kinematics: (T, 180)
    skel_features = np.concatenate([pos_flat, vel_flat, acc_flat], axis=1)

    # 3. Early Fusion
    # Concatenate Audio + Skeleton: (T, 193)
    features = np.concatenate([audio_norm, skel_features], axis=1)

    return torch.tensor(features, dtype=torch.float32)


def sliding_window_inference(model, features, device):
    """
    Performs sliding window inference on a full sequence.

    Args:
        model (nn.Module): Trained model
        features (torch.Tensor): Full sequence features (T, InputDim)
        device (torch.device): Device to run on

    Returns:
        np.ndarray: Frame-wise probabilities (T, NumClasses)
    """
    model.eval()
    seq_len = features.shape[0]
    num_classes = Config.NUM_CLASSES
    window_size = Config.WINDOW_SIZE
    stride = int(window_size * (1 - Config.INFERENCE_OVERLAP))

    # Buffers for aggregation
    probs_sum = torch.zeros((seq_len, num_classes), device=device)
    counts = torch.zeros((seq_len, 1), device=device)

    # Handle short sequences
    if seq_len < window_size:
        # Pad to window size
        pad_len = window_size - seq_len
        feat_padded = torch.nn.functional.pad(
            features, (0, 0, 0, pad_len), mode="constant", value=0
        )
        input_tensor = feat_padded.unsqueeze(0).to(device)  # (1, W, D)

        with torch.no_grad():
            _, _, logits3 = model(input_tensor)
            probs = torch.softmax(logits3, dim=2).squeeze(0)  # (W, C)

        # Add valid part
        probs_sum[:seq_len] += probs[:seq_len]
        counts[:seq_len] += 1

    else:
        # Sliding window
        # Prepare batch of windows for efficiency?
        # For simplicity and memory safety with variable lengths, we process one by one or small batches.
        # Given the constraints, processing window by window is safe.

        start_indices = list(range(0, seq_len - window_size + 1, stride))
        # Ensure last frame is covered
        if start_indices[-1] != seq_len - window_size:
            start_indices.append(seq_len - window_size)

        for start_idx in start_indices:
            end_idx = start_idx + window_size
            window = features[start_idx:end_idx]
            input_tensor = window.unsqueeze(0).to(device)

            with torch.no_grad():
                _, _, logits3 = model(input_tensor)
                probs = torch.softmax(logits3, dim=2).squeeze(0)

            probs_sum[start_idx:end_idx] += probs
            counts[start_idx:end_idx] += 1

    # Average
    avg_probs = probs_sum / (counts + 1e-8)
    return avg_probs.cpu().numpy()


def decode_predictions(probs):
    """
    Decodes frame-wise probabilities into a list of gesture IDs.
    Applies RLE and Minimum Duration Filtering.

    Args:
        probs (np.ndarray): (T, NumClasses)

    Returns:
        list: List of gesture IDs (int)
    """
    # 1. Argmax
    labels = np.argmax(probs, axis=1)

    # 2. Run-Length Encoding
    # groupby returns (key, group_iterator)
    rle = [(k, len(list(g))) for k, g in itertools.groupby(labels)]

    # 3. Filter
    final_gestures = []
    for label, duration in rle:
        # Ignore background (0)
        if label == Config.BACKGROUND_LABEL:
            continue

        # Min Duration Filter
        if duration >= Config.MIN_GESTURE_DURATION:
            final_gestures.append(int(label))

    return final_gestures


def run_inference(load_cached_data=True):
    """
    Main function to run inference on validation and test sets.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}")

    # 1. Load Data and Stats
    # We use the data loader functions to get raw lists
    print("Loading data...")
    train_raw = load_and_process_split(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_raw = load_and_process_split(Config.VAL_METADATA_PATH, "val", load_cached_data)
    test_raw = load_and_process_split(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    stats = compute_normalization_stats(train_raw, load_cached_data)

    # 2. Load Model
    print("Loading model...")
    model = HNGKN().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    else:
        print(
            f"Warning: Model file not found at {Config.MODEL_SAVE_PATH}. Inference will use random weights."
        )

    # 3. Validation Evaluation
    print("Evaluating on Validation Set...")
    val_preds = []
    val_gts = []

    # Process validation samples
    # val_raw['labels'] contains frame-wise labels (numpy arrays)
    # We need to convert frame-wise labels back to gesture list for GT comparison
    # Or rely on the 'labels' field in metadata if we had it handy, but here we have frame arrays.
    # We can perform RLE on the GT frame array to get the GT sequence.

    for i in range(len(val_raw["ids"])):
        audio = val_raw["audio"][i]
        skel = val_raw["skeleton"][i]
        gt_frames = val_raw["labels"][i]

        # Preprocess
        features = preprocess_sequence(audio, skel, stats)

        # Inference
        probs = sliding_window_inference(model, features, device)

        # Decode Prediction
        pred_seq = decode_predictions(probs)
        val_preds.append(pred_seq)

        # Decode GT (RLE on frame labels)
        # Note: GT labels in raw data are already 0 for background and 1-20 for gestures.
        # We just need to extract the sequence.
        gt_rle = [(k, len(list(g))) for k, g in itertools.groupby(gt_frames)]
        gt_seq = [int(k) for k, d in gt_rle if k != Config.BACKGROUND_LABEL]
        val_gts.append(gt_seq)

    # Compute Metric
    val_score = compute_normalized_levenshtein(val_preds, val_gts)
    print(f"Validation Normalized Levenshtein Distance: {val_score}")

    # 4. Test Submission
    print("Generating Test Submission...")
    submission_lines = []

    for i in range(len(test_raw["ids"])):
        sample_id = test_raw["ids"][i]
        audio = test_raw["audio"][i]
        skel = test_raw["skeleton"][i]

        # Preprocess
        features = preprocess_sequence(audio, skel, stats)

        # Inference
        probs = sliding_window_inference(model, features, device)

        # Decode
        pred_seq = decode_predictions(probs)

        # Format: SessionID,Label1,Label2,...
        if len(pred_seq) > 0:
            pred_str = ",".join(map(str, pred_seq))
            line = f"{sample_id},{pred_str}"
        else:
            # If no gestures detected, just the ID (or ID, with nothing)
            # Based on format "Session00001,2,12,3", if empty -> "Session00001"
            line = f"{sample_id}"

        submission_lines.append(line)

    # Save to file
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    with open(Config.SUBMISSION_PATH, "w") as f:
        for line in submission_lines:
            f.write(line + "\n")

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
