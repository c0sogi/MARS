import os
import torch
import pandas as pd
import numpy as np
from library.config import (
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    SUBMISSION_DIR,
    DEVICE,
)
from library.utils import get_device, seed_everything
from library.data_loader import get_dataloaders
from library.model import HRVANet


def predict_and_submit(load_cached_data=True):
    """
    Loads the trained model, generates predictions for the test set,
    and saves the results to a CSV file in the required submission format.

    Args:
        load_cached_data (bool): Whether to load pre-processed data arrays from cache.
                                 Defaults to True.
    """
    # 1. Setup
    seed_everything()
    device = get_device()

    # Ensure submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print(f"Starting inference on device: {device}")

    # 2. Load Data
    # We only need the test_loader, but get_dataloaders returns all three
    _, _, test_loader = get_dataloaders(
        TRAIN_META_PATH,
        VAL_META_PATH,
        TEST_META_PATH,
        load_cached_data=load_cached_data,
    )

    if test_loader is None:
        print(f"Error: Test data could not be loaded from {TEST_META_PATH}.")
        return

    # 3. Load Model
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}. Cannot predict.")
        return

    model = HRVANet().to(device)

    try:
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()

    # 4. Inference Loop
    all_probs = []
    all_ids = []

    print("Running prediction loop...")
    with torch.no_grad():
        for inputs, patient_ids in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)

            # Apply Sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits)

            # Store results
            # Flatten probs to 1D array
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_ids.extend(list(patient_ids))

    # 5. Generate Submission
    if not all_ids:
        print("Warning: No predictions generated. Test set might be empty.")
        return

    # Convert IDs to integers as per sample_submission.csv format (int64)
    # The dataloader returns them as strings (e.g., "00001")
    formatted_ids = [int(pid) for pid in all_ids]

    submission_df = pd.DataFrame({"BraTS21ID": formatted_ids, "MGMT_value": all_probs})

    # Save to CSV
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print("First 5 predictions:")
    print(submission_df.head())
