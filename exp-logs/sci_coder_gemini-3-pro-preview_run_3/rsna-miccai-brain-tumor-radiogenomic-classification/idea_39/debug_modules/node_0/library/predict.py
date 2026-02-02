import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.model import SSFNet


def generate_submission(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Generates the submission file for the test set using the trained SSFNet model.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        batch_size (int): Batch size for inference.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Inference Device: {device}")

    # 2. Load Data
    # get_dataloaders returns (train, val, test). We only need test.
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 3. Load Model
    print(f"Loading model from {Config.MODEL_PATH}...")
    model = SSFNet()

    if os.path.exists(Config.MODEL_PATH):
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {Config.MODEL_PATH}")

    model.to(device)
    model.eval()

    # 4. Inference
    print("Running inference on test set...")
    all_probs = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for (even, odd), _ in test_loader:
            # Move inputs to device
            even = even.to(device)
            odd = odd.to(device)

            # Forward pass
            logits = model(even, odd)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            all_probs.extend(probs)

    # 5. Retrieve IDs
    # The test_loader.dataset has the 'ids' attribute.
    # Since shuffle=False for test_loader, the order matches.
    test_ids = test_loader.dataset.ids

    if len(test_ids) != len(all_probs):
        raise ValueError(
            f"Mismatch between number of IDs ({len(test_ids)}) and predictions ({len(all_probs)})"
        )

    # 6. Create Submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": all_probs})

    # 7. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
