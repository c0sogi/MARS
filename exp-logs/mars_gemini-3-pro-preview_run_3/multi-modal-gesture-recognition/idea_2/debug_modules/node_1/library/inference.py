import torch
import numpy as np
import os
from typing import List, Tuple, Optional

from library.config import Config
from library.model import MSTCN
from library.dataset import create_dataloaders
from library.utils import (
    load_checkpoint,
    write_submission_file,
    seed_everything,
)


def decode_predictions(frame_preds: np.ndarray) -> List[int]:
    """
    Decodes frame-wise predictions into a list of gesture IDs.
    Applies Run-Length Encoding, removes background class (0),
    and filters out segments shorter than 5 frames.

    Args:
        frame_preds: Numpy array of shape (T,) containing class indices.

    Returns:
        List of integer gesture IDs.
    """
    if len(frame_preds) == 0:
        return []

    # 1. Run-Length Encoding
    segments = []
    current_label = frame_preds[0]
    current_len = 1

    for label in frame_preds[1:]:
        if label == current_label:
            current_len += 1
        else:
            segments.append((current_label, current_len))
            current_label = label
            current_len = 1
    segments.append((current_label, current_len))

    # 2. Filter Background and Short Segments
    final_gestures = []
    for label, length in segments:
        # Remove background class (ID 0)
        if label != Config.BACKGROUND_CLASS_ID:
            # Remove extremely short segments (< 5 frames)
            if length >= 5:
                final_gestures.append(int(label))

    return final_gestures


def predict_sequence(
    model: torch.nn.Module, features: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Runs the model on a batch of sequences.

    Args:
        model: The trained MS-TCN model.
        features: Input features tensor of shape (Batch, Time, Input_Dim).
        mask: Boolean mask tensor of shape (Batch, Time).

    Returns:
        Logits tensor from the final stage of shape (Batch, Classes, Time).
    """
    # Forward pass returns a list of outputs for each stage
    outputs = model(features, mask)

    # We use the output of the final refinement stage for prediction
    final_stage_logits = outputs[-1]

    return final_stage_logits


def run_inference(debug_subset_size: Optional[int] = None):
    """
    Main inference routine. Loads the best model, generates predictions
    for the test set, and writes the submission file.

    Args:
        debug_subset_size: If provided, limits the test set size for debugging.
    """
    # Ensure reproducibility
    seed_everything(Config.RANDOM_SEED)
    device = Config.DEVICE

    print(f"Initializing inference on device: {device}")

    # 1. Initialize Model
    model = MSTCN().to(device)

    # 2. Load Checkpoint
    checkpoint_path = Config.MODEL_SAVE_PATH
    if os.path.exists(checkpoint_path):
        try:
            model, _, epoch, loss = load_checkpoint(model, None, checkpoint_path)
            print(
                f"Loaded model checkpoint from {checkpoint_path} (Epoch {epoch}, Val Loss {loss:.4f})"
            )
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights."
        )

    model.eval()

    # 3. Create Test DataLoader
    # We only need the test loader here
    _, _, test_loader = create_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug_subset_size=debug_subset_size,
    )

    all_predictions: List[Tuple[str, List[int]]] = []

    print("Generating predictions...")

    with torch.no_grad():
        for features, _, mask, lengths, ids in test_loader:
            features = features.to(device)
            mask = mask.to(device)

            # Run model
            logits = predict_sequence(model, features, mask)

            # Get predicted class indices: (Batch, Time)
            pred_classes = torch.argmax(logits, dim=1)

            # Process each sample in the batch
            for i in range(len(ids)):
                sample_id = ids[i]
                length = lengths[i]

                # Extract valid frames (ignore padding)
                # pred_classes[i] is (Max_Time,), we slice [:length]
                valid_preds = pred_classes[i, :length].cpu().numpy()

                # Decode into gesture list
                decoded_gestures = decode_predictions(valid_preds)

                all_predictions.append((sample_id, decoded_gestures))

    # 4. Write Submission
    output_path = Config.SUBMISSION_PATH
    print(f"Writing predictions to {output_path}...")
    write_submission_file(all_predictions, output_path)
    print("Inference complete.")
