import os
import torch
import numpy as np
import pandas as pd
from library import config, model, features, data_utils

# ==========================================
# Reproducibility
# ==========================================
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)


def predict_sequence(model_instance, skeleton, audio, stats, device):
    """
    Performs sliding window inference on a single sequence.

    Args:
        model_instance: Trained PyTorch model.
        skeleton: np.ndarray (T, 20, 3)
        audio: np.ndarray (T, 13)
        stats: Normalization statistics dict.
        device: torch.device

    Returns:
        probs: np.ndarray (T, NumClasses) - Frame-wise probabilities.
    """
    model_instance.eval()

    # 1. Feature Extraction & Fusion
    # augment=False for deterministic inference
    fused_features = features.process_sample(skeleton, audio, stats, augment=False)
    # Shape: (T, InputDim)

    seq_len = fused_features.shape[0]
    window_size = config.WINDOW_SIZE
    stride = window_size // 2  # 50% overlap

    # buffers for aggregation
    prob_buffer = np.zeros((seq_len, config.NUM_CLASSES), dtype=np.float32)
    count_buffer = np.zeros((seq_len, config.NUM_CLASSES), dtype=np.float32)

    # 2. Handle Short Sequences (Padding)
    if seq_len < window_size:
        # Pad with zeros
        pad_len = window_size - seq_len
        padded_feat = np.pad(fused_features, ((0, pad_len), (0, 0)), mode="constant")

        input_tensor = (
            torch.from_numpy(padded_feat).unsqueeze(0).to(device)
        )  # (1, Window, Feat)

        with torch.no_grad():
            outputs = model_instance(input_tensor)
            # Use Stage 3 probabilities
            probs = outputs["probs_3"].cpu().numpy()[0]  # (Window, Classes)

        # Trim padding
        return probs[:seq_len]

    # 3. Sliding Window Loop
    windows = []
    indices = []

    # Generate windows
    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        windows.append(fused_features[start:end])
        indices.append((start, end))

    # Handle the final window if it doesn't align perfectly with stride
    if seq_len > window_size and indices[-1][1] < seq_len:
        start = seq_len - window_size
        end = seq_len
        windows.append(fused_features[start:end])
        indices.append((start, end))

    # Batch processing could be done here, but for simplicity/memory we process one by one
    # or small batches. Given the constraints, simple loop is fine.

    with torch.no_grad():
        for i, window_feat in enumerate(windows):
            start, end = indices[i]

            input_tensor = torch.from_numpy(window_feat).unsqueeze(0).to(device)
            outputs = model_instance(input_tensor)

            # Get Stage 3 probabilities
            win_probs = outputs["probs_3"].cpu().numpy()[0]  # (Window, Classes)

            # Accumulate
            prob_buffer[start:end] += win_probs
            count_buffer[start:end] += 1.0

    # 4. Average
    # Avoid division by zero (though count should be >= 1)
    count_buffer[count_buffer == 0] = 1.0
    final_probs = prob_buffer / count_buffer

    return final_probs


def decode_predictions(probs):
    """
    Decodes frame-wise probabilities into a sequence of gesture IDs.
    Applies Argmax -> Run-Length Encoding -> Background Filtering.

    Args:
        probs: np.ndarray (T, NumClasses)

    Returns:
        List of integer gesture IDs.
    """
    # 1. Argmax
    frame_labels = np.argmax(probs, axis=1)

    # 2. Run-Length Encoding (Collapse consecutive duplicates)
    if len(frame_labels) == 0:
        return []

    collapsed_labels = [frame_labels[0]]
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != frame_labels[i - 1]:
            collapsed_labels.append(frame_labels[i])

    # 3. Filter Background (Class 0)
    final_sequence = [
        lbl for lbl in collapsed_labels if lbl != config.BACKGROUND_CLASS_ID
    ]

    return final_sequence


def generate_submission(load_cached_data=True):
    """
    Main inference routine. Loads model, predicts on test set, writes submission.
    """
    device = torch.device(config.DEVICE)
    print(f"Inference device: {device}")

    # 1. Load Data & Stats
    # We only strictly need test data and stats here
    _, _, test_data, stats = features.load_data_and_stats(
        load_cached_data=load_cached_data
    )

    # 2. Load Model
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {config.MODEL_SAVE_PATH}. Train first."
        )

    weskn_model = model.WESKN().to(device)
    state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    weskn_model.load_state_dict(state_dict)
    weskn_model.eval()
    print("Model loaded successfully.")

    # 3. Prediction Loop
    results = []

    skeletons = test_data["skeleton"]
    audios = test_data["audio"]
    sample_ids = test_data["sample_ids"]

    print(f"Generating predictions for {len(sample_ids)} test samples...")

    for i, sample_id in enumerate(sample_ids):
        skel = skeletons[i]
        aud = audios[i]

        # Predict
        probs = predict_sequence(weskn_model, skel, aud, stats, device)

        # Decode
        pred_ids = decode_predictions(probs)

        # Format: "ID1,ID2,ID3"
        # Note: The submission format requires comma separation
        pred_str = ",".join(map(str, pred_ids))

        results.append({"Id": sample_id, "Predicted": pred_str})

    # 4. Write Submission
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # The required format is: SessionID,label1,label2,...
    # Example: Session00001,2,12,3
    # We will write this manually to ensure exact formatting

    with open(config.SUBMISSION_PATH, "w") as f:
        for res in results:
            line = f"{res['Id']},{res['Predicted']}\n"
            f.write(line)

    print(f"Submission saved to {config.SUBMISSION_PATH}")
