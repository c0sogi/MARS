import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.trainer import SHCGKN
from library.data_loader import get_data_loaders
from library.utils import decode_predictions


def generate_submission(load_cached_data=True):
    """
    Generates predictions for the test set using the trained SHC-GKN model.
    Performs sliding window inference with temporal ensembling (averaging overlapping windows).
    Saves the result to Config.SUBMISSION_PATH.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed dataset from cache.
    """
    # 1. Setup
    device = Config.get_device()
    print(f"Running inference on device: {device}")

    # 2. Load Data
    # We only need the test loader
    _, _, test_loader = get_data_loaders(
        config=Config, load_cached_data=load_cached_data
    )
    dataset = test_loader.dataset

    # 3. Load Model
    model = SHCGKN(Config).to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading model weights from {Config.MODEL_SAVE_PATH}")
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.MODEL_SAVE_PATH}. Using random weights."
        )

    model.eval()

    # 4. Inference Loop (Sliding Window Aggregation)
    # The dataset flattens all sequences. We need to reconstruct the global timeline.
    # dataset.all_labels has the shape (TotalFrames,) - initialized to 0s for test, but correct length.
    total_frames = dataset.all_labels.shape[0] if dataset.all_labels is not None else 0

    if total_frames == 0:
        print("Error: Test dataset is empty.")
        return

    # Global accumulators for temporal ensembling
    global_probs = np.zeros((total_frames, Config.NUM_CLASSES), dtype=np.float32)
    global_counts = np.zeros((total_frames,), dtype=np.float32)

    print("Processing batches...")
    with torch.no_grad():
        for batch_idx, (features, _) in enumerate(test_loader):
            features = features.to(device)

            # Forward pass
            outputs = model(features)

            # Use Stage 3 probabilities for final prediction
            probs = outputs["probs3"].cpu().numpy()  # Shape: (Batch, Time, Classes)

            # Map batch windows back to global timeline
            start_window_idx = batch_idx * test_loader.batch_size

            for i in range(features.size(0)):
                window_idx = start_window_idx + i

                # Safety check
                if window_idx >= len(dataset.windows):
                    break

                # Get global start/end indices for this window
                # These indices point to the raw concatenated arrays in the dataset
                global_start, global_end = dataset.windows[window_idx]

                # Determine valid length
                # The window in the dataset might be shorter than Config.WINDOW_SIZE (e.g. at end of clip)
                # The __getitem__ method pads it to Config.WINDOW_SIZE.
                # We only want to accumulate the valid (non-padded) part.
                valid_len = global_end - global_start
                pred_len = probs.shape[1]  # Should be Config.WINDOW_SIZE

                actual_len = min(valid_len, pred_len)

                # Accumulate probabilities
                global_probs[global_start : global_start + actual_len] += probs[
                    i, :actual_len
                ]
                global_counts[global_start : global_start + actual_len] += 1.0

    # 5. Normalization
    # Average the probabilities where windows overlapped
    mask = global_counts > 0
    global_probs[mask] /= global_counts[mask, None]

    # 6. Decoding and Formatting
    print("Decoding sequences...")

    # Load test metadata to get Sample IDs in the correct order
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    if Config.DEBUG:
        test_meta = test_meta.head(Config.DEBUG_SUBSET_SIZE)

    results = []

    # Iterate over samples using the sample_indices stored in the dataset
    # These correspond 1-to-1 with the rows in test_meta
    for idx, (start, end) in enumerate(dataset.sample_indices):
        # Get Sample ID
        sample_id = test_meta.iloc[idx]["sample_id"]

        # Extract probabilities for this specific sequence
        sample_probs = global_probs[start:end]

        # Decode: Probs -> Argmax -> RLE -> Filter -> Sequence
        pred_seq = decode_predictions(sample_probs)

        # Format: "SampleID,Label1,Label2,..."
        pred_str = ",".join(map(str, pred_seq))
        results.append(f"{sample_id},{pred_str}")

    # 7. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    with open(Config.SUBMISSION_PATH, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    # This block is not required by the prompt but useful for local testing
    generate_submission()
