import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, log_message, get_device
from library.data_loader import get_dataloaders
from library.model import MGSHDNetwork


def run_inference(load_cached_data=True):
    """
    Runs inference on the test set and generates the submission file.

    Args:
        load_cached_data (bool): Whether to use cached pre-processed data.
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = get_device()
    log_message(f"Inference using device: {device}")

    # 2. Load Data
    log_message("Loading test data...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    if test_loader is None:
        log_message("Error: Test data loader is None. Cannot proceed with inference.")
        return

    # 3. Load Model
    log_message("Initializing model...")
    model = MGSHDNetwork().to(device)

    if os.path.exists(Config.MODEL_PATH):
        log_message(f"Loading weights from {Config.MODEL_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        log_message(
            f"Warning: Model checkpoint not found at {Config.MODEL_PATH}. Using random weights."
        )

    # 4. Inference
    model.eval()
    all_ids = []
    all_probs = []

    log_message("Starting inference loop...")
    with torch.no_grad():
        for inputs, _, ids in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            # Collect results
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_ids.extend(ids)

    # 5. Generate Submission
    log_message(f"Generating submission for {len(all_ids)} subjects...")

    submission_df = pd.DataFrame({"BraTS21ID": all_ids, "MGMT_value": all_probs})

    # Ensure output directory exists (handled by Config.setup, but double check for file write)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    log_message(f"Submission saved to {Config.SUBMISSION_PATH}")
