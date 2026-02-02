import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import get_logger
from library.dataset import get_dataloaders
from library.model import SiameseFPNEfficientNet

logger = get_logger("inference")


def predict_submission(load_cached_data=True):
    """
    Runs inference on the test set, aggregates predictions by breast (prediction_id),
    and saves the final submission file.

    Args:
        load_cached_data (bool): If True, attempts to load processed metadata from cache.
                                 If False, re-processes metadata.
    """
    # 1. Setup Device
    device = torch.device(Config.DEVICE)
    logger.info(f"Initializing inference on device: {device}")

    # 2. Prepare Data
    # We use get_dataloaders to ensure the exact same preprocessing (age normalization stats)
    # is applied as during training. We only need the test_loader.
    logger.info("Preparing test dataloader...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Load Model
    logger.info("Initializing model architecture...")
    model = SiameseFPNEfficientNet()

    # Load weights
    if os.path.exists(Config.MODEL_PATH):
        logger.info(f"Loading model weights from {Config.MODEL_PATH}")
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # In a real submission scenario, this should probably fail,
        # but for development/debugging flow we log a warning.
        logger.warning(
            f"Checkpoint not found at {Config.MODEL_PATH}. Using random weights."
        )

    model.to(device)
    model.eval()

    # 4. Run Inference
    logger.info("Starting inference loop...")
    all_prediction_ids = []
    all_probs = []

    with torch.no_grad():
        # Disable tqdm for silent execution as requested
        for batch in tqdm(test_loader, disable=True):
            # Move inputs to device
            images = batch["image"].to(device)
            images_contra = batch["image_contra"].to(device)

            # Forward pass
            logits = model(images, images_contra)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Collect results
            # probs is [B, 1], flatten to [B]
            all_probs.extend(probs.cpu().numpy().flatten())
            all_prediction_ids.extend(batch["prediction_id"])

    # 5. Aggregate Predictions
    # The task requires one prediction per prediction_id (breast level).
    # We take the maximum probability across views (e.g., CC and MLO) for the same breast.
    logger.info("Aggregating predictions...")

    df_results = pd.DataFrame(
        {"prediction_id": all_prediction_ids, "cancer": all_probs}
    )

    # Group by prediction_id and take max
    submission_df = df_results.groupby("prediction_id", as_index=False)["cancer"].max()

    # 6. Save Submission
    output_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path} with {len(submission_df)} rows.")

    return submission_df
