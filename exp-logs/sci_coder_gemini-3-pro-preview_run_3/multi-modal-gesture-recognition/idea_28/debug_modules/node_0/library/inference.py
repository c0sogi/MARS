import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import load_dataset, compute_kinematics
from library.model import RGHC_MN


def decode_predictions(frame_probs):
    """
    Decodes frame-wise probabilities into a list of gesture IDs.
    Applies Argmax, RLE, Background Suppression, and Min Duration Filtering.
    """
    # 1. Argmax to get raw labels
    # frame_probs: (T, NumClasses)
    raw_labels = np.argmax(frame_probs, axis=1)

    # 2. Run-Length Encoding
    segments = []
    if len(raw_labels) > 0:
        current_label = raw_labels[0]
        current_len = 1

        for l in raw_labels[1:]:
            if l == current_label:
                current_len += 1
            else:
                segments.append((current_label, current_len))
                current_label = l
                current_len = 1
        segments.append((current_label, current_len))

    # 3. Filter and Collect
    final_sequence = []
    for label, length in segments:
        # Filter Background (0) and Short Segments (< 5 frames)
        if label != 0 and length >= Config.MIN_GESTURE_DURATION:
            final_sequence.append(int(label))

    return final_sequence


def predict_sequence(model, skeleton, audio, device):
    """
    Performs sliding window inference on a single sequence.
    """
    model.eval()

    # 1. Preprocess Features (Full Sequence)
    # skeleton: (T, 20, 3) -> kinematics: (T, 20, 9)
    kinematics = compute_kinematics(skeleton)
    T, J, C = kinematics.shape

    # Flatten: (T, 180)
    skel_flat = kinematics.reshape(T, J * C)

    # Concatenate Audio: (T, 193)
    features = np.concatenate([skel_flat, audio], axis=1)

    # 2. Setup Sliding Windows
    window_size = Config.WINDOW_SIZE
    stride = Config.STRIDE_TEST  # 32

    # Buffers for aggregation
    probs_sum = np.zeros((T, Config.NUM_CLASSES), dtype=np.float32)
    counts = np.zeros((T, 1), dtype=np.float32)

    # Prepare batches
    windows_list = []
    indices_list = []

    # Handle short sequences
    if T < window_size:
        # Pad to window size
        pad_len = window_size - T
        feat_pad = np.zeros((pad_len, features.shape[1]), dtype=features.dtype)
        window = np.concatenate([features, feat_pad], axis=0)
        windows_list.append(window)
        indices_list.append((0, T))  # Valid range
    else:
        # Sliding window
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            window = features[start:end]
            windows_list.append(window)
            indices_list.append((start, end))

        # Handle tail
        if T > window_size and (T - window_size) % stride != 0:
            start = T - window_size
            end = T
            window = features[start:end]
            windows_list.append(window)
            indices_list.append((start, end))

    if not windows_list:
        return np.zeros((T, Config.NUM_CLASSES))

    # Convert to tensor
    windows_tensor = torch.tensor(np.array(windows_list), dtype=torch.float32).to(
        device
    )

    # 3. Inference
    with torch.no_grad():
        # Process in batches to avoid OOM on huge sequences (though usually 1 seq is fine)
        batch_size = Config.BATCH_SIZE
        num_windows = len(windows_list)

        for i in range(0, num_windows, batch_size):
            batch_input = windows_tensor[i : i + batch_size]

            # Forward pass
            outputs = model(batch_input)

            # Get Stage 3 probabilities
            batch_probs = outputs["stage3"].cpu().numpy()  # (B, Window, Classes)

            # Aggregate
            current_indices = indices_list[i : i + batch_size]
            for j, (start, end) in enumerate(current_indices):
                # If sequence was shorter than window, we padded input.
                # The output is (WindowSize, Classes).
                # We only care about the valid part.
                valid_len = end - start

                # If it was a short sequence padded
                if T < window_size:
                    p = batch_probs[j, :valid_len, :]
                else:
                    p = batch_probs[j, :, :]

                probs_sum[start:end] += p
                counts[start:end] += 1.0

    # 4. Average
    # Avoid division by zero
    counts[counts == 0] = 1.0
    avg_probs = probs_sum / counts

    return avg_probs


def generate_submission(load_cached_data=True):
    """
    Main function to generate submission file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading test data...")
    data = load_dataset(
        Config.TEST_METADATA_PATH,
        Config.TEST_CACHE_PATH,
        load_cached_data=load_cached_data,
    )

    skeletons = data["skeletons"]
    audio = data["audio"]
    sample_ids = data["sample_ids"]

    print(f"Loaded {len(sample_ids)} test sequences.")

    # 2. Load Model
    print("Loading model...")
    model = RGHC_MN().to(device)

    if os.path.exists(Config.BEST_MODEL_PATH):
        checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        print("Model weights loaded successfully.")
    else:
        print(f"Error: Model checkpoint not found at {Config.BEST_MODEL_PATH}")
        return

    # 3. Generate Predictions
    results = []
    print("Starting inference...")

    for i, (skel, aud, sid) in enumerate(zip(skeletons, audio, sample_ids)):
        # Predict
        probs = predict_sequence(model, skel, aud, device)

        # Decode
        pred_seq = decode_predictions(probs)

        # Format string: "Label1,Label2,..."
        if not pred_seq:
            # If empty, leave blank after comma? Or just ID?
            # Task description: "Session00001,2,12,3"
            # If no gestures, likely just "Session00001" or "Session00001,"
            # Usually empty string is safer if no gestures found.
            pred_str = ""
        else:
            pred_str = ",".join(map(str, pred_seq))

        results.append((sid, pred_str))

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(sample_ids)} sequences.")

    # 4. Save Submission
    print("Saving submission...")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # We need to write manually to match format "SessionID,Label1,..."
    # Pandas to_csv might add quotes or headers we don't want if not careful.
    # The example is: "Session00001,2,12,3"

    with open(Config.SUBMISSION_PATH, "w") as f:
        for sid, pred_str in results:
            if pred_str:
                line = f"{sid},{pred_str}\n"
            else:
                line = f"{sid}\n"  # No gestures predicted
            f.write(line)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
