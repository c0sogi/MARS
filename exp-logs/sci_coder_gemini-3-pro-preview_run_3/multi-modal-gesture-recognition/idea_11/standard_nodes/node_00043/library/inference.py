import os
import torch
import numpy as np
from library.config import Config
from library.model import BA_AKN
from library.data_loader import get_dataloaders
from library.utils import setup_logger, collapse_predictions


def run_inference(load_cached_data=True):
    """
    Executes the inference pipeline:
    1. Loads the best trained model.
    2. Processes the test dataset via sliding windows.
    3. Aggregates predictions (Temporal Ensembling).
    4. Generates the submission CSV.
    """
    logger = setup_logger(os.path.join(Config.WORKING_DIR, "inference.log"))
    device = Config.DEVICE

    logger.info("Initializing Inference Pipeline...")

    # 1. Load Data
    # We only need the test loader and ids.
    # get_dataloaders returns: train_loader, val_loader, test_loader, test_ids
    _, _, test_loader, test_ids = get_dataloaders(
        load_cached_data=load_cached_data, debug=Config.DEBUG
    )

    # Access the underlying dataset to get raw sequence info for reconstruction
    dataset = test_loader.dataset
    raw_skeletons = dataset.skeletons  # List of (T, 20, 3) arrays

    # 2. Load Model
    model = BA_AKN().to(device)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if os.path.exists(model_path):
        logger.info(f"Loading model weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        logger.warning(
            f"Model checkpoint not found at {model_path}. Using random initialization (Expect poor results)."
        )

    model.eval()

    # 3. Initialize Buffers for Temporal Ensembling
    # seq_probs: Dict[seq_idx] -> np.array (T, NumClasses)
    # seq_counts: Dict[seq_idx] -> np.array (T,)
    seq_probs = {}
    seq_counts = {}

    for idx, skel in enumerate(raw_skeletons):
        T = len(skel)
        seq_probs[idx] = np.zeros((T, Config.NUM_CLASSES), dtype=np.float32)
        seq_counts[idx] = np.zeros((T,), dtype=np.float32)

    logger.info("Starting Sliding Window Inference...")

    current_window_idx = 0

    with torch.no_grad():
        for features, _, _ in test_loader:
            # features shape: (Batch, Channels, Time) or (Batch, Time, Channels) depending on loader
            # Based on data_loader.py, it returns (Batch, InputDim, Time) if permuted?
            # Checking GestureDataset: returns torch.FloatTensor(features). Features is (W, InputDim).
            # So DataLoader yields (Batch, W, InputDim).
            # Model expects (Batch, Time, InputDim) -> BiGRUEncoder handles this.

            features = features.to(device)
            batch_size = features.size(0)

            # Forward Pass
            outputs = model(features)

            # Get Stage 3 Class Logits (Batch, Classes, Time)
            logits = outputs["stage3_cls"]

            # Apply Softmax to get probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (Batch, Classes, Time)

            # Map windows back to sequences
            for b in range(batch_size):
                global_idx = current_window_idx + b
                if global_idx >= len(dataset.windows):
                    break

                seq_idx, start_frame = dataset.windows[global_idx]

                # Transpose probs to (Time, Classes) for easier slicing
                window_probs = probs[b].transpose(1, 0)

                # Determine valid range in the original sequence
                # The window covers [start_frame : start_frame + Config.WINDOW_SIZE]
                # We must clip to the actual sequence length
                seq_len = len(raw_skeletons[seq_idx])
                end_frame = min(start_frame + Config.WINDOW_SIZE, seq_len)

                valid_len = end_frame - start_frame

                if valid_len > 0:
                    # Accumulate probabilities
                    seq_probs[seq_idx][start_frame:end_frame] += window_probs[
                        :valid_len
                    ]
                    seq_counts[seq_idx][start_frame:end_frame] += 1.0

            current_window_idx += batch_size

    # 4. Decode and Generate Submission
    logger.info("Decoding sequences and generating submission CSV...")

    results = []

    for seq_idx in range(len(test_ids)):
        sample_id = test_ids[seq_idx]

        # Average probabilities
        counts = seq_counts[seq_idx][:, None]
        # Avoid division by zero for frames that might not have been covered (shouldn't happen with correct stride)
        counts[counts == 0] = 1.0

        avg_probs = seq_probs[seq_idx] / counts

        # Argmax to get frame labels
        frame_preds = np.argmax(avg_probs, axis=1)

        # Collapse predictions (RLE + Background Removal)
        collapsed_preds = collapse_predictions(frame_preds)

        # Format: SessionID,label1,label2,...
        pred_str = ",".join(map(str, collapsed_preds))
        results.append(f"{sample_id},{pred_str}")

    # Write to file
    output_path = Config.SUBMISSION_FILE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for line in results:
            f.write(line + "\n")

    logger.info(f"Submission saved to {output_path}")
