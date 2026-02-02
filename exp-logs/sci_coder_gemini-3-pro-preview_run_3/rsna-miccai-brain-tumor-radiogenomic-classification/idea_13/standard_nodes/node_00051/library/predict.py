import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from library import config
from library import utils
from library import data_loader
from library import model


def run_prediction(load_cached_data=True, max_samples=None):
    """
    Runs inference on the test set and generates the submission file.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        max_samples (int, optional): Limit the number of samples for debugging.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = utils.get_device()
    print(f"Device: {device}")

    # 2. Load Test Data
    # We use get_data_for_split directly to avoid loading train/val data into memory
    print("Loading test data...")
    test_X, _, test_ids = data_loader.get_data_for_split(
        "test",
        config.TEST_META_PATH,
        load_cached_data=load_cached_data,
        max_samples=max_samples,
    )

    # Create TensorDataset and DataLoader
    # Note: test_X is numpy float32, converted to torch tensor
    test_dataset = TensorDataset(torch.from_numpy(test_X))

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print(f"Test Data Loaded: {len(test_dataset)} samples")

    # 3. Load Model
    net = model.MGMTNet()
    net = net.to(device)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {config.MODEL_SAVE_PATH}. Please train the model first."
        )

    print(f"Loading model weights from {config.MODEL_SAVE_PATH}...")
    state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    # 4. Inference
    print("Starting inference...")
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            # TensorDataset returns a tuple (tensor,), so we unpack it
            inputs = batch[0].to(device)

            # Forward pass
            logits = net(inputs)

            # Apply sigmoid to get probabilities (0-1)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())

    # Concatenate results
    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs).flatten()
    else:
        all_probs = np.array([])

    # 5. Generate Submission
    # Ensure IDs and Predictions align
    if len(test_ids) != len(all_probs):
        print(
            f"Warning: Number of IDs ({len(test_ids)}) does not match number of predictions ({len(all_probs)})"
        )

    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": all_probs})

    # Ensure output directory exists
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    print(f"Saving submission to {config.SUBMISSION_PATH}...")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)

    print("Prediction complete.")
    return submission_df
