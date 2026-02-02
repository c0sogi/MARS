import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_data, KinematicAugmentor
from library.model import KC_IRN
from library.train import decode_predictions


def predict_sequence(model, sample, device):
    """
    Performs sliding window inference on a single test sequence.

    Args:
        model (nn.Module): The trained KC_IRN model.
        sample (dict): Dictionary containing 'skeleton' and 'audio' data.
        device (torch.device): Device to run inference on.

    Returns:
        list: Ordered list of predicted gesture IDs.
    """
    window_size = Config.WINDOW_SIZE
    stride = Config.TEST_STRIDE
    num_classes = Config.NUM_CLASSES

    # 1. Feature Engineering
    # Extract raw skeleton and compute kinematics (Pos, Vel, Acc)
    raw_skel = sample["skeleton"]  # (T, 20, 3)
    kinematics = KinematicAugmentor.compute_kinematics(raw_skel)  # (T, 180)

    # Extract audio
    audio = sample["audio"]  # (T, 13)

    # Concatenate for Early Fusion
    full_features = np.concatenate([kinematics, audio], axis=1)  # (T, 193)
    T = full_features.shape[0]

    # 2. Initialize Buffers for Probability Accumulation
    prob_buffer = np.zeros((T, num_classes), dtype=np.float32)
    count_buffer = np.zeros((T, 1), dtype=np.float32)

    # 3. Sliding Window Loop
    # Handle sequences shorter than window size
    if T < window_size:
        pad_len = window_size - T
        # Pad features: edge padding for temporal continuity
        feat_padded = np.pad(full_features, ((0, pad_len), (0, 0)), mode="edge")

        feat_tensor = torch.from_numpy(feat_padded).float().unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(feat_tensor)
            # Use the output from the final refinement stage
            final_logits = outputs[-1]
            probs = (
                torch.softmax(final_logits, dim=2).cpu().numpy().squeeze(0)
            )  # (Window, Classes)

        # Accumulate only the valid part
        prob_buffer += probs[:T]
        count_buffer += 1.0

    else:
        # Standard sliding window
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            window_feat = full_features[start:end]

            feat_tensor = torch.from_numpy(window_feat).float().unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(feat_tensor)
                final_logits = outputs[-1]
                probs = torch.softmax(final_logits, dim=2).cpu().numpy().squeeze(0)

            prob_buffer[start:end] += probs
            count_buffer[start:end] += 1.0

        # Handle the final window if it wasn't covered exactly by the stride
        last_start = T - window_size
        if last_start > 0 and (last_start % stride != 0):
            window_feat = full_features[last_start:T]

            feat_tensor = torch.from_numpy(window_feat).float().unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(feat_tensor)
                final_logits = outputs[-1]
                probs = torch.softmax(final_logits, dim=2).cpu().numpy().squeeze(0)

            prob_buffer[last_start:T] += probs
            count_buffer[last_start:T] += 1.0

    # 4. Average Probabilities and Decode
    # Avoid division by zero (though count_buffer should be >= 1 everywhere)
    count_buffer[count_buffer == 0] = 1.0
    avg_probs = prob_buffer / count_buffer

    # Dense frame-wise predictions
    pred_dense = np.argmax(avg_probs, axis=1)

    # Decode to sparse gesture list (RLE + remove background)
    pred_seq = decode_predictions(pred_dense)

    return pred_seq


def generate_submission(load_cached_data=True):
    """
    Generates the submission file for the test set.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # Setup
    set_seed()
    device = torch.device(Config.DEVICE)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Load Data
    print("Loading test data...")
    test_data = load_data("test", load_cached_data=load_cached_data)

    # Load Model
    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Train the model first."
        )

    model = KC_IRN().to(device)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Generate Predictions
    print("Generating predictions...")
    submission_lines = []

    for sample in test_data:
        sample_id = sample["id"]

        # Predict
        predicted_labels = predict_sequence(model, sample, device)

        # Format: SessionID,label1,label2,...
        # If list is empty, just SessionID (or SessionID, but usually implies no gestures)
        label_str = ",".join(map(str, predicted_labels))
        line = f"{sample_id},{label_str}" if label_str else f"{sample_id},"

        submission_lines.append(line)

    # Save Submission
    with open(Config.SUBMISSION_PATH, "w") as f:
        for line in submission_lines:
            f.write(line + "\n")

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
