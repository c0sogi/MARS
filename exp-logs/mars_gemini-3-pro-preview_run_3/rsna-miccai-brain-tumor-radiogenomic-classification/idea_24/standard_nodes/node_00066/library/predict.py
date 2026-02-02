import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_device
from library.model import RMSHDNet
from library.data_loader import get_dataloaders


def generate_submission():
    """
    Loads the best trained model, performs inference on the test set,
    and generates the submission.csv file.
    """
    # 1. Setup
    device = get_device()
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    print(f"Inference Device: {device}")

    # 2. Load Data
    # We only need the test loader and the test IDs.
    # load_cached_data=True ensures we use the cache if available (created during training or previous runs)
    _, _, test_loader, test_ids = get_dataloaders(load_cached_data=True)

    # 3. Load Model
    model = RMSHDNet().to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading model weights from {Config.MODEL_SAVE_PATH}...")
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"Model weights not found at {Config.MODEL_SAVE_PATH}. Train the model first."
        )

    model.eval()

    # 4. Inference
    all_probs = []

    print(f"Starting inference on {len(test_ids)} test samples...")

    with torch.no_grad():
        for data in test_loader:
            # Move inputs to device
            data = data.to(device)

            # Forward pass
            logits = model(data)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Move to CPU and numpy
            probs_np = probs.cpu().numpy().flatten()
            all_probs.extend(probs_np)

    # 5. Format Submission
    # The IDs in test_ids are strings (e.g., "00001").
    # The submission format typically expects integers for BraTS21ID (e.g., 1).
    # We convert them here.
    formatted_ids = [int(pid) for pid in test_ids]

    submission_df = pd.DataFrame({"BraTS21ID": formatted_ids, "MGMT_value": all_probs})

    # Ensure output directory exists
    os.makedirs(submission_dir, exist_ok=True)

    # Save
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(submission_df.head())
