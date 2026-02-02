import os
import numpy as np
import torch
import torch.nn.functional as F
from library.config import (
    NUM_CLASSES,
    WINDOW_SIZE,
    INFERENCE_STRIDE,
    MIN_DURATION,
    SUBMISSION_DIR,
    SEED,
    WORKING_DIR,
)
from library.utils import run_length_encoding
from library.data_loader import get_test_loader, KinematicAugmentor
from library.model import ASH_KN

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def sliding_window_inference(model, skeleton, audio, device):
    """
    Performs sliding window inference on a single sequence with temporal averaging.

    Args:
        model: Trained ASH_KN model.
        skeleton: (T, J, 3) numpy array of raw skeleton data.
        audio: (T, MFCC) numpy array of audio features.
        device: torch device.

    Returns:
        np.ndarray: Frame-wise class probabilities (T, NumClasses).
    """
    model.eval()
    num_frames = skeleton.shape[0]

    # Prepare Augmentor (No augmentation for test, just feature derivation)
    augmentor = KinematicAugmentor(augment=False)

    # Pre-calculate features for the whole sequence
    # KinematicAugmentor expects (T, J, 3) -> returns (T, J*9) [Pos, Vel, Acc]
    kinematic_feats = augmentor(skeleton)

    # Concatenate with Audio
    # audio is (T, MFCC)
    # full_features: (T, InputDim)
    full_features = np.concatenate([kinematic_feats, audio], axis=1)

    # Prepare for accumulation of probabilities
    accumulated_probs = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
    count_matrix = np.zeros((num_frames, 1), dtype=np.float32)

    # Sliding Window Configuration
    step = INFERENCE_STRIDE
    win_size = WINDOW_SIZE

    windows = []
    indices = []

    # Handle sequences shorter than the window size by padding
    if num_frames < win_size:
        pad_len = win_size - num_frames
        padded_feats = np.pad(full_features, ((0, pad_len), (0, 0)), mode="constant")
        windows.append(padded_feats)
        indices.append((0, num_frames))  # Valid range in the padded window
    else:
        # Generate overlapping windows
        for start in range(0, num_frames - win_size + 1, step):
            end = start + win_size
            windows.append(full_features[start:end])
            indices.append((start, end))

        # Handle the final frame if not covered perfectly by stride
        if (num_frames - win_size) % step != 0:
            start = num_frames - win_size
            end = num_frames
            windows.append(full_features[start:end])
            indices.append((start, end))

    # Batch processing
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch_wins = windows[i : i + batch_size]
            batch_idxs = indices[i : i + batch_size]

            # Convert to Tensor: (Batch, Time, Feats)
            input_tensor = torch.FloatTensor(np.array(batch_wins)).to(device)

            # Forward pass through the ASH-KN model
            # Returns logits from all three stages: logits1, logits2, logits3
            _, _, logits3 = model(input_tensor)

            # Apply Softmax to the final stage output
            probs3 = torch.softmax(logits3, dim=2).cpu().numpy()

            # Accumulate probabilities into the global buffer
            for j, (start, end) in enumerate(batch_idxs):
                if num_frames < win_size:
                    # Special case for short sequence:
                    # The window is padded, but we only accumulate valid frames
                    accumulated_probs[0:num_frames] += probs3[j, 0:num_frames]
                    count_matrix[0:num_frames] += 1
                else:
                    accumulated_probs[start:end] += probs3[j]
                    count_matrix[start:end] += 1

    # Compute average probabilities
    # Avoid division by zero using maximum(count, 1.0)
    avg_probs = accumulated_probs / np.maximum(count_matrix, 1.0)

    return avg_probs


def post_process_predictions(frame_probs):
    """
    Converts frame-wise probabilities to a list of gesture IDs.
    Applies Argmax, Run-Length Encoding, and Duration Filtering.

    Args:
        frame_probs: (T, NumClasses) numpy array of probabilities.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    # 1. Argmax to get class indices for each frame
    predictions = np.argmax(frame_probs, axis=1)

    # 2. Run-Length Encoding with Min Duration Filter
    # This collapses consecutive frames and removes segments shorter than MIN_DURATION
    gesture_list = run_length_encoding(predictions, min_duration=MIN_DURATION)

    return gesture_list


def generate_submission(model_path, submission_filename="submission.csv"):
    """
    Generates the final submission CSV for the test set.

    Args:
        model_path: Path to the trained model weights (.pth file).
        submission_filename: Name of the output CSV file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and Load Model
    model = ASH_KN().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Error: Model file not found at {model_path}")
        return

    model.eval()

    # Retrieve Test Data
    # We use get_test_loader to get the raw data dictionary (cached)
    # batch_size is irrelevant here as we access data by key
    _, test_data = get_test_loader(batch_size=1, load_cached=True)

    results = []

    # Process samples in deterministic order
    sample_ids = sorted(test_data.keys())

    print(f"Generating predictions for {len(sample_ids)} test samples...")

    for sid in sample_ids:
        sample = test_data[sid]
        skel = sample["skeleton"]
        audio = sample["audio"]

        # Handle cases where data might be missing or empty
        if skel is None or len(skel) == 0:
            pred_seq = []
        else:
            # 1. Inference
            avg_probs = sliding_window_inference(model, skel, audio, device)

            # 2. Post-processing
            pred_seq = post_process_predictions(avg_probs)

        # Format output: SessionID,Label1,Label2,...
        if len(pred_seq) > 0:
            pred_str = ",".join(map(str, pred_seq))
            line = f"{sid},{pred_str}"
        else:
            line = f"{sid},"  # Empty prediction

        results.append(line)

    # Save to CSV
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, submission_filename)

    with open(out_path, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Submission saved to {out_path}")
