import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from library import config, model, utils, data_loader


def generate_predictions():
    """
    Runs inference on the test dataset using the trained GHCMN model.
    Generates a submission file with recognized gesture sequences.
    """
    # Set device
    device = config.DEVICE
    print(f"Using device: {device}")

    # 1. Load Test Data
    # We manually load only the test set to avoid the overhead of loading train/val metadata
    test_cache = os.path.join(config.CACHE_DIR, "dataset_test.npz")

    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    print("Loading test data...")
    test_data = data_loader.load_and_cache_data(
        config.TEST_METADATA_PATH, test_cache, load_cached_data=True
    )

    # Create Dataset and Loader
    # Stride is set in config.STRIDE (32) which provides 50% overlap with WINDOW_SIZE (64)
    test_dataset = data_loader.GestureDataset(
        test_data, mode="test", window_size=config.WINDOW_SIZE, stride=config.STRIDE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model
    print("Initializing model...")
    net = model.KC_IRN().to(device)

    # 3. Load Weights
    if os.path.exists(config.BEST_MODEL_PATH):
        print(f"Loading weights from {config.BEST_MODEL_PATH}")
        state_dict = torch.load(config.BEST_MODEL_PATH, map_location=device)
        net.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {config.BEST_MODEL_PATH}. Predictions will be random."
        )

    net.eval()

    # 4. Inference Loop
    print("Running inference...")

    # Buffers for sequence reconstruction
    # Map sample_idx (int) -> tensor of shape (SeqLen, NumClasses)
    seq_preds = {}
    seq_counts = {}

    # Initialize buffers based on dataset metadata
    # dataset.ids is a list of sample_id strings
    # dataset.skeletons is a list of skeleton arrays (T, 20, 3)
    # We use the integer index in dataset.ids as the key
    num_samples = len(test_dataset.ids)
    for i in range(num_samples):
        seq_len = test_dataset.skeletons[i].shape[0]
        seq_preds[i] = torch.zeros((seq_len, config.NUM_CLASSES), device=device)
        seq_counts[i] = torch.zeros((seq_len, 1), device=device)

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            features = features.to(device)

            # Forward pass
            outputs = net(features)

            # We use Stage 3 probabilities for final prediction (Deep Supervision refinement)
            probs = outputs["stage3_probs"]  # (B, T, C)

            # Map batch items back to original sequences
            start_idx = batch_idx * config.BATCH_SIZE

            for i in range(features.size(0)):
                global_idx = start_idx + i

                # Retrieve window metadata from dataset
                # dataset.windows is a list of (sample_idx, start_frame)
                sample_idx, start_frame = test_dataset.windows[global_idx]

                # Determine valid window length
                # The loader pads if sequence < window_size
                seq_len = seq_preds[sample_idx].shape[0]
                window_len = config.WINDOW_SIZE

                # Calculate end frame, clamping to the actual sequence length
                end_frame = min(start_frame + window_len, seq_len)
                valid_len = end_frame - start_frame

                if valid_len > 0:
                    # Slice valid probabilities
                    # If padding was added by loader, we ignore the padded part of the output
                    w_probs = probs[i, :valid_len, :]

                    # Accumulate probabilities and counts for averaging
                    seq_preds[sample_idx][start_frame:end_frame] += w_probs
                    seq_counts[sample_idx][start_frame:end_frame] += 1

    # 5. Decode and Save
    print("Decoding predictions...")
    submission_rows = []

    for i in range(num_samples):
        sample_id = test_dataset.ids[i]

        # Average probabilities
        # Clamp count to 1 to avoid division by zero (though count >= 1 for all valid frames)
        avg_probs = seq_preds[i] / seq_counts[i].clamp(min=1)

        # Argmax to get frame-wise labels
        pred_labels = torch.argmax(avg_probs, dim=1).cpu().numpy()

        # Post-process:
        # 1. Run-Length Encoding
        # 2. Filter segments shorter than MIN_GESTURE_DURATION
        # 3. Merge adjacent segments
        # 4. Remove BACKGROUND_CLASS_ID
        pred_seq = utils.process_gesture_sequence(
            pred_labels,
            min_duration=config.MIN_GESTURE_DURATION,
            background_id=config.BACKGROUND_CLASS_ID,
        )

        # Format: SessionID,label1,label2,...
        if not pred_seq:
            label_str = ""
        else:
            label_str = ",".join(map(str, pred_seq))

        submission_rows.append(f"{sample_id},{label_str}")

    # Write to file
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    with open(config.SUBMISSION_PATH, "w") as f:
        for row in submission_rows:
            f.write(row + "\n")

    print(f"Submission saved to {config.SUBMISSION_PATH}")
