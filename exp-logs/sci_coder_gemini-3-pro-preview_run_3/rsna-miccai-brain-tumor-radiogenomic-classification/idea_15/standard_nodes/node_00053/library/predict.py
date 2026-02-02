import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_logger, get_device
from library.data_loader import get_dataloaders
from library.model import MGMT25DModel

logger = get_logger()


def generate_submission(model_path=None, load_cached_data=True):
    """
    Loads the best model, runs inference on the test set, and generates submission.csv.

    Args:
        model_path (str, optional): Path to the trained model weights.
                                    Defaults to 'best_model.pth' in the cache directory.
        load_cached_data (bool): Whether to load pre-processed data from cache.
                                 Defaults to True.
    """
    # 1. Setup Device and Paths
    device = get_device()

    if model_path is None:
        model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        logger.error(
            f"Model file not found at {model_path}. Cannot generate submission."
        )
        return

    # 2. Prepare Data
    # We use get_dataloaders to ensure consistent preprocessing with training.
    # It returns (train_loader, val_loader, test_loader). We only need the third one.
    logger.info("Initializing DataLoaders for inference...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Load Model
    logger.info(f"Loading model from {model_path}...")
    model = MGMT25DModel().to(device)

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Failed to load state dict: {e}")
        return

    model.eval()

    # 4. Inference Loop
    predictions = []
    ids = []

    logger.info("Starting inference on test set...")
    with torch.no_grad():
        for inputs, pids in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Handle pids (can be tensor or tuple depending on DataLoader/Dataset behavior)
            if isinstance(pids, torch.Tensor):
                pids = pids.numpy()

            predictions.extend(probs)
            ids.extend(pids)

    # 5. Create Submission DataFrame
    df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Ensure BraTS21ID is integer format (e.g., 00013 -> 13)
    df["BraTS21ID"] = df["BraTS21ID"].astype(int)

    # 6. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df.to_csv(Config.SUBMISSION_FILE, index=False)

    logger.info(f"Inference complete. Submission saved to {Config.SUBMISSION_FILE}")
    logger.info(f"Total predictions generated: {len(df)}")
