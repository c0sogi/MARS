import os
import numpy as np
import pandas as pd
import torch

from library.config import Config, set_seed
from library.utils import get_logger
from library.model import AsymmetricEfficientNet
from library.data_loader import get_dataloaders

# Initialize logger
logger = get_logger(name="Inference")


def predict_with_tta(model, images, device):
    """
    Generates predictions using Test-Time Augmentation (TTA).
    Augmentations: Original, Horizontal Flip, Vertical Flip.
    Returns the average probability across views.
    """
    # 1. Original
    outputs_orig = model(images)
    probs_orig = torch.sigmoid(outputs_orig)

    # 2. Horizontal Flip (dim 3 is width)
    images_h = torch.flip(images, [3])
    outputs_h = model(images_h)
    probs_h = torch.sigmoid(outputs_h)

    # 3. Vertical Flip (dim 2 is height)
    images_v = torch.flip(images, [2])
    outputs_v = model(images_v)
    probs_v = torch.sigmoid(outputs_v)

    # Average probabilities
    avg_probs = (probs_orig + probs_h + probs_v) / 3.0
    return avg_probs


def generate_submission(load_cached_data=True):
    """
    Generates the submission file for the test set.
    Implements caching for predictions to allow rapid regeneration of CSVs
    without re-running the neural network inference.

    Args:
        load_cached_data (bool): If True, attempts to load raw predictions from disk.
    """
    set_seed(Config.SEED)

    # Ensure working directory exists for cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, "test_predictions.npy")

    # Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA):
        logger.error(f"Test metadata not found at {Config.TEST_METADATA}")
        return

    test_df = pd.read_csv(Config.TEST_METADATA)
    logger.info(f"Loaded test metadata: {len(test_df)} samples.")

    predictions = None

    # --------------------------------------------------------------------------
    # 1. Try Loading Cache
    # --------------------------------------------------------------------------
    if load_cached_data and os.path.exists(cache_path):
        try:
            logger.info(f"Loading cached predictions from {cache_path}...")
            predictions = np.load(cache_path)
            if len(predictions) != len(test_df):
                logger.warning(
                    "Cached predictions length mismatch. Re-running inference."
                )
                predictions = None
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Re-running inference.")
            predictions = None

    # --------------------------------------------------------------------------
    # 2. Run Inference (if no cache)
    # --------------------------------------------------------------------------
    if predictions is None:
        logger.info("Starting inference process...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference device: {device}")

        # Initialize Model
        model = AsymmetricEfficientNet()

        # Load Weights
        if not os.path.exists(Config.MODEL_PATH):
            logger.error(
                f"Model weights not found at {Config.MODEL_PATH}. Cannot proceed."
            )
            return

        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully.")

        # Get DataLoader
        # Note: get_dataloaders handles anchor caching internally.
        loaders = get_dataloaders(test_df=test_df)
        test_loader = loaders["test"]

        preds_list = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)

                # Predict with TTA
                batch_probs = predict_with_tta(model, images, device)

                # Move to CPU and store
                preds_list.append(batch_probs.cpu().numpy())

        # Concatenate all batches
        predictions = np.concatenate(preds_list, axis=0).flatten()

        # Save to cache
        try:
            np.save(cache_path, predictions)
            logger.info(f"Predictions saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save prediction cache: {e}")

    # --------------------------------------------------------------------------
    # 3. Create Submission File
    # --------------------------------------------------------------------------
    # The DataLoader with shuffle=False preserves the order of test_df
    submission_df = pd.DataFrame(
        {"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": predictions}
    )

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission file saved to {Config.SUBMISSION_PATH}")

    # Log sample
    logger.info("Sample predictions:")
    logger.info(submission_df.head().to_string())

    return submission_df
