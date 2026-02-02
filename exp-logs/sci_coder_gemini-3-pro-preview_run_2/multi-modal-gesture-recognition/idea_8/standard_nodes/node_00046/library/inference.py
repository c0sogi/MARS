import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Import library modules
from library import config
from library import utils
from library import data_loader
from library import model


def load_model(checkpoint_path, device):
    """
    Initializes the model and loads weights from the checkpoint.
    """
    # Initialize architecture
    net = model.DSR_CRCN()
    net = net.to(device)

    # Load weights
    if os.path.exists(checkpoint_path):
        # print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    net.eval()
    return net


def generate_submission(load_cached_data=True):
    """
    Main inference function. Loads data, model, generates predictions, and saves submission.
    """
    utils.set_seed(config.SEED)
    device = utils.get_device()

    # 1. Load Data
    # We only need the test loader here.
    # Note: get_data_loaders returns (train, val, test).
    _, _, test_loader = data_loader.get_data_loaders(load_cached_data=load_cached_data)

    # 2. Load Model
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    try:
        net = load_model(checkpoint_path, device)
    except FileNotFoundError:
        print("Error: Best model checkpoint not found. Cannot generate submission.")
        return

    # 3. Inference Loop
    results = []

    with torch.no_grad():
        for batch in test_loader:
            ids = batch["ids"]
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"]

            # Forward pass
            # Returns logits0, logits1, logits2
            _, _, logits2 = net(features, mask)

            # Process each sample in the batch
            for i, session_id in enumerate(ids):
                # Get valid length for this sequence
                seq_len = lengths[i].item()

                # Extract logits for valid frames only: [Time, NumClasses]
                # logits2 is [Batch, Time, NumClasses]
                sample_logits = logits2[i, :seq_len, :]

                # Convert to probabilities
                sample_probs = F.softmax(sample_logits, dim=1).cpu().numpy()

                # 4. Post-Processing (Smoothing)
                # Use Stage 2 output as final prediction stream
                smoothed_probs = utils.smooth_predictions(
                    sample_probs, window_size=config.MEDIAN_WINDOW
                )

                # 5. Decoding
                predicted_gestures = utils.decode_sequence(smoothed_probs)

                # Format for submission
                # "SessionID,Label1,Label2,..."
                label_str = ",".join(map(str, predicted_gestures))
                results.append((session_id, label_str))

    # 6. Save Submission
    # Sort by Session ID to ensure consistent order (optional but good practice)
    results.sort(key=lambda x: x[0])

    # Ensure submission directory exists
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    with open(config.SUBMISSION_PATH, "w") as f:
        for session_id, label_str in results:
            if label_str:
                f.write(f"{session_id},{label_str}\n")
            else:
                # Handle empty predictions (no gestures detected)
                # The format implies SessionID, then labels. If empty, just SessionID?
                # Or SessionID, (trailing comma)?
                # Based on example: "Session00001,2,12,3"
                # If empty, we'll write "Session00001," or just "Session00001"
                # Let's write just the ID to be safe, or ID with empty string
                f.write(f"{session_id},\n")

    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(f"Generated predictions for {len(results)} sequences.")
