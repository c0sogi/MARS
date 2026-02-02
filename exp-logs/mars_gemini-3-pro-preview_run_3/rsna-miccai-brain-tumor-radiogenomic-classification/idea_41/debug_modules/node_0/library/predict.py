import os
import torch
import pandas as pd
import numpy as np

from library.config import (
    MODEL_SAVE_PATH,
    SUBMISSION_DIR,
    SUBMISSION_PATH,
    BATCH_SIZE,
    seed_everything,
)
from library.utils import get_device
from library.data_loader import get_dataloaders
from library.model import SiameseEfficientNet


def generate_submission(load_cached_data=True):
    """
    Loads the best trained model, runs inference on the test set,
    and saves the predictions to a CSV file.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
                                 Defaults to True.
    """
    # 1. Setup
    seed_everything()
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    # We only need the test loader and the corresponding IDs
    print("Loading test data...")
    _, _, test_loader, test_ids = get_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = SiameseEfficientNet()
    model.to(device)

    # 4. Load Weights
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading weights from {MODEL_SAVE_PATH}...")
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Model file not found at {MODEL_SAVE_PATH}.")
        print("Using initialized weights (random predictions).")

    # 5. Inference
    model.eval()
    all_probs = []

    print(f"Starting inference on {len(test_ids)} test samples...")
    with torch.no_grad():
        for inputs in test_loader:
            # Unpack inputs (BraTSDataset returns (x_even, x_odd) when y is None)
            x_even, x_odd = inputs

            # Move to device
            x_even = x_even.to(device)
            x_odd = x_odd.to(device)

            # Forward pass
            logits = model(x_even, x_odd)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Collect results
            all_probs.extend(probs.cpu().numpy().flatten())

    # 6. Generate Submission DataFrame
    # Ensure exact alignment between IDs and Predictions
    if len(test_ids) != len(all_probs):
        raise ValueError(
            f"Mismatch between number of IDs ({len(test_ids)}) and predictions ({len(all_probs)})."
        )

    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": all_probs})

    # 7. Save to CSV
    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print("Submission generated successfully.")
    print("First 5 rows:")
    print(submission_df.head())
