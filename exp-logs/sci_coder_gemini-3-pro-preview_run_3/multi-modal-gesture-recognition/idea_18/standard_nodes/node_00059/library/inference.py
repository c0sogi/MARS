import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.utils import set_seed, decode_predictions_to_sequence
from library.data_loader import GestureDataset
from library.model import (
    AKCIRN,
    INPUT_DIM,
    NUM_CLASSES,
    HIDDEN_DIM,
    WINDOW_SIZE,
    STRIDE,
    DEVICE,
)


def predict_sliding_window(model, dataloader, device):
    """
    Performs inference using a sliding window approach with temporal ensembling.

    Args:
        model (nn.Module): The trained AKCIRN model.
        dataloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run inference on.

    Returns:
        tuple: (test_probs, test_counts)
            test_probs (dict): Mapping of sample_id -> accumulated probabilities (T, C).
            test_counts (dict): Mapping of sample_id -> overlap counts (T,).
    """
    model.eval()
    dataset = dataloader.dataset

    # Initialize buffers
    test_probs = {}
    test_counts = {}

    # Pre-allocate based on dataset samples
    for s in dataset.samples:
        sid = s["sample_id"]
        # Use the raw skeleton length as the sequence length
        t = s["skeleton"].shape[0]
        test_probs[sid] = np.zeros((t, NUM_CLASSES), dtype=np.float32)
        test_counts[sid] = np.zeros((t,), dtype=np.float32)

    with torch.no_grad():
        for i, (features, _) in enumerate(dataloader):
            features = features.to(device)

            # Forward pass: Get Stage 3 logits (l3)
            _, _, l3 = model(features)

            # Apply softmax to get probabilities
            probs = F.softmax(l3, dim=1).cpu().numpy()  # (B, C, T)

            # Map batch items back to original samples
            batch_size = features.size(0)
            start_idx = i * dataloader.batch_size

            for b in range(batch_size):
                global_idx = start_idx + b
                if global_idx >= len(dataset.windows):
                    break

                # Retrieve window metadata
                sample_idx, start_frame = dataset.windows[global_idx]
                sample_data = dataset.samples[sample_idx]
                sid = sample_data["sample_id"]

                # Transpose probabilities to (T, C) for easier slicing
                p = probs[b].transpose(1, 0)

                # Determine valid length (handle padding at edges)
                actual_len = sample_data["skeleton"].shape[0]
                valid_len = min(WINDOW_SIZE, actual_len - start_frame)

                if valid_len > 0:
                    # Accumulate probabilities and counts
                    test_probs[sid][start_frame : start_frame + valid_len] += p[
                        :valid_len
                    ]
                    test_counts[sid][start_frame : start_frame + valid_len] += 1.0

    return test_probs, test_counts


def decode_predictions(test_probs, test_counts):
    """
    Decodes accumulated probabilities into gesture sequences.

    Args:
        test_probs (dict): Accumulated probabilities.
        test_counts (dict): Overlap counts.

    Returns:
        dict: Mapping of sample_id -> list of gesture IDs.
    """
    predictions = {}

    for sid, prob_sum in test_probs.items():
        counts = test_counts[sid]

        # Avoid division by zero
        counts[counts == 0] = 1.0

        # Temporal Averaging
        avg_probs = prob_sum / counts[:, None]

        # Argmax to get frame-wise labels
        pred_labels = np.argmax(avg_probs, axis=1)

        # Decode to sequence using RLE (Run-Length Encoding)
        # This function handles collapsing duplicates and filtering background (0)
        seq = decode_predictions_to_sequence(pred_labels)
        predictions[sid] = seq

    return predictions


def run_inference(
    base_dir="./",
    cache_dir="./working/idea_18",
    model_path="./working/best_model.pth",
    submission_dir="./submission",
    batch_size=32,
):
    """
    Main driver function to run inference and generate submission.

    Args:
        base_dir (str): Base directory containing metadata.
        cache_dir (str): Directory for caching dataset files.
        model_path (str): Path to the trained model checkpoint.
        submission_dir (str): Directory to save the submission CSV.
        batch_size (int): Batch size for inference.
    """
    set_seed(42)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    print(f"Initializing Inference...")
    print(f"Model Path: {model_path}")

    # 1. Load Test Dataset
    test_meta = os.path.join(base_dir, "metadata/test.csv")

    # Note: augment=False for inference
    test_ds = GestureDataset(
        test_meta,
        split="test",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        cache_dir=cache_dir,
        load_cached=True,
        augment=False,
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 2. Load Model
    model = AKCIRN(
        input_dim=INPUT_DIM, num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model weights loaded successfully.")
    else:
        print(f"WARNING: Model file not found at {model_path}. Using random weights.")

    # 3. Predict
    print("Running sliding window inference...")
    probs, counts = predict_sliding_window(model, test_loader, DEVICE)

    # 4. Decode
    print("Decoding predictions...")
    predictions = decode_predictions(probs, counts)

    # 5. Save Submission
    submission_path = os.path.join(submission_dir, "submission.csv")
    print(f"Saving submission to {submission_path}...")

    with open(submission_path, "w") as f:
        # Iterate based on dataset sample order to ensure consistency
        for s in test_ds.samples:
            sid = s["sample_id"]
            seq = predictions.get(sid, [])

            # Format: SessionID,label1,label2,...
            seq_str = ",".join(map(str, seq))
            f.write(f"{sid},{seq_str}\n")

    print("Inference complete.")
