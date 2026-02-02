import os
import torch
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.utils import set_seed, setup_logger, rle_encode_predictions
from library.data_loader import get_dataloaders
from library.model import RDKRN


def run_inference(limit_samples=None):
    """
    Executes the inference pipeline:
    1. Loads the trained RDKRN model.
    2. Processes the test dataset using sliding windows.
    3. Aggregates probabilities (Temporal Ensembling).
    4. Decodes sequences (Argmax + RLE).
    5. Saves predictions to CSV.

    Args:
        limit_samples (int, optional): Limit the number of test samples for debugging.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    logger = setup_logger("Inference", os.path.join(Config.WORK_DIR, "inference.log"))
    logger.info(f"Initializing Inference on device: {device}")

    # 2. Data Loading
    # We only need the test loader
    _, _, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, limit_samples=limit_samples
    )

    # 3. Model Loading
    model = RDKRN().to(device)
    model_path = Config.MODEL_SAVE_PATH

    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found at {model_path}. Cannot proceed.")
        return

    logger.info(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Prepare Aggregation Buffers
    # We need to reconstruct full sequences from sliding windows.
    # Access dataset info to pre-allocate buffers for each sequence.
    dataset = test_loader.dataset
    buffers = {}

    # seq_info is a list of (global_start_idx, length, sample_id)
    # We use sample_id as the key.
    for _, length, sample_id in dataset.seq_info:
        buffers[sample_id] = {
            "probs": np.zeros((length, Config.NUM_CLASSES), dtype=np.float32),
            "counts": np.zeros((length, 1), dtype=np.float32),
        }

    logger.info(f"Prepared buffers for {len(buffers)} sequences.")

    # 5. Inference Loop
    logger.info("Starting sliding window inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            features = batch["features"].to(device)
            sample_ids = batch["sample_id"]
            frame_starts = batch["frame_start"]

            # Forward Pass
            # Returns list of outputs from [Stage1, Stage2, Stage3]
            outputs = model(features)

            # Use Stage 3 (Refinement 2) logits for final prediction
            stage3_logits = outputs[2]

            # Convert to probabilities
            probs = F.softmax(stage3_logits, dim=2).cpu().numpy()

            # Aggregate into buffers
            for i, sid in enumerate(sample_ids):
                if sid not in buffers:
                    continue

                start = frame_starts[i]
                p = probs[i]  # Shape: (WindowSize, NumClasses)

                # Determine valid range in the buffer
                # Buffer size is the actual sequence length
                buffer_len = buffers[sid]["probs"].shape[0]
                window_len = p.shape[0]

                # Calculate end index in buffer
                end = start + window_len

                # Clip to buffer length (handles padding in the last window or short sequences)
                buffer_end = min(end, buffer_len)

                # Calculate valid length from the window
                valid_len = buffer_end - start

                if valid_len > 0:
                    # Add probabilities and increment counts
                    buffers[sid]["probs"][start:buffer_end] += p[:valid_len]
                    buffers[sid]["counts"][start:buffer_end] += 1.0

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    # 6. Decoding and Formatting
    logger.info("Decoding sequences...")
    results = []

    # Sort by sample_id to ensure deterministic order
    sorted_sids = sorted(buffers.keys())

    for sid in sorted_sids:
        data = buffers[sid]

        # Average probabilities
        # Avoid division by zero (though coverage should be complete)
        counts = data["counts"]
        counts[counts == 0] = 1.0
        avg_probs = data["probs"] / counts

        # Argmax to get frame labels
        pred_labels = np.argmax(avg_probs, axis=1)

        # Run-Length Encoding (collapses duplicates and removes background)
        sequence = rle_encode_predictions(
            pred_labels, background_id=Config.BACKGROUND_CLASS_ID
        )

        # Format as comma-separated string
        seq_str = ",".join(map(str, sequence))

        results.append((sid, seq_str))

    # 7. Save Submission
    submission_path = Config.SUBMISSION_PATH
    logger.info(f"Saving submission to {submission_path}")

    try:
        with open(submission_path, "w") as f:
            for sid, seq_str in results:
                # Format: SessionID,label1,label2,...
                if seq_str:
                    line = f"{sid},{seq_str}"
                else:
                    line = f"{sid}"  # Empty sequence
                f.write(line + "\n")
        logger.info("Submission saved successfully.")

    except Exception as e:
        logger.error(f"Failed to save submission: {e}")
