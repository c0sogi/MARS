import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    MODEL_SAVE_PATH,
    SUBMISSION_FILE_PATH,
    SEED,
    WORKING_DIR,
)
from library.utils import seed_everything, get_logger
from library.data_loader import get_datasets
from library.model_arch import MNSHDNetwork

logger = get_logger("inference")


def generate_submission(
    model_path=MODEL_SAVE_PATH, output_path=SUBMISSION_FILE_PATH, load_cached_data=True
):
    """
    Generates predictions for the test set using the trained model and saves them to a CSV file.

    Args:
        model_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV.
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    # 1. Setup
    seed_everything(SEED)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Initializing inference pipeline...")

    # 2. Data Loading
    # Use the library function which handles caching logic internally.
    # get_datasets returns (train, val, test). We only need the test dataset.
    logger.info(f"Loading test data (Cached: {load_cached_data})...")
    _, _, test_dataset = get_datasets(load_cached_data=load_cached_data)

    # Handle empty dataset case
    if len(test_dataset) == 0:
        logger.warning("Test dataset is empty. Generating empty submission file.")
        pd.DataFrame(columns=["BraTS21ID", "MGMT_value"]).to_csv(
            output_path, index=False
        )
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Must be False to preserve ID alignment
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    # 3. Model Initialization
    logger.info(f"Loading model architecture and weights from {model_path}...")
    model = MNSHDNetwork().to(DEVICE)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully.")
    else:
        logger.error(f"Model file not found at {model_path}. Cannot perform inference.")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.eval()

    # 4. Inference Loop
    all_probs = []

    logger.info(f"Starting prediction on {len(test_dataset)} samples...")

    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            # BraTSDataset returns only images for the test set (y is None)
            images = images.to(DEVICE)

            # Forward pass
            logits = model(images)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Move to CPU and store
            all_probs.append(probs.cpu().numpy())

    # 5. Post-processing
    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs).flatten()
    else:
        all_probs = np.array([])

    # Retrieve IDs from the dataset (order is preserved since shuffle=False)
    all_ids = test_dataset.ids

    # 6. Save Submission
    submission_df = pd.DataFrame({"BraTS21ID": all_ids, "MGMT_value": all_probs})

    logger.info(f"Saving submission to {output_path}...")
    submission_df.to_csv(output_path, index=False)

    logger.info("Submission generation complete.")
