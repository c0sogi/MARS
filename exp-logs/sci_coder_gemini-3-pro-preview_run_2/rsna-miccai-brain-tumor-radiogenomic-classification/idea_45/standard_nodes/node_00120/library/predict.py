import os
import torch
import pandas as pd
import logging
from library.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    SEED,
)
from library.utils import setup_logger, seed_everything, ensure_dir
from library.data_loader import get_dataloaders
from library.model import AsymmetricEfficientNet, predict_with_tta

logger = setup_logger("predict")


def generate_submission(debug=False, max_samples=None):
    """
    Generates the submission file using the trained model and Test-Time Augmentation (TTA).

    Args:
        debug (bool): If True, runs on a subset of the data.
        max_samples (int): Maximum number of samples to process.
    """
    # 1. Setup
    seed_everything(SEED)
    ensure_dir(os.path.dirname(SUBMISSION_PATH))

    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    # We only need the test loader for prediction
    logger.info("Initializing Test DataLoader...")
    _, _, test_loader = get_dataloaders(debug=debug, max_samples=max_samples)

    # 3. Model Initialization
    logger.info("Initializing AsymmetricEfficientNet...")
    model = AsymmetricEfficientNet().to(device)

    # 4. Load Weights
    if os.path.exists(MODEL_SAVE_PATH):
        logger.info(f"Loading model weights from {MODEL_SAVE_PATH}...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        logger.warning(
            f"Model file not found at {MODEL_SAVE_PATH}. Using random weights (expect poor performance)."
        )

    # 5. Inference with TTA
    # predict_with_tta is imported from library.model as requested
    logger.info("Running inference with TTA...")
    test_ids, preds = predict_with_tta(model, test_loader, device)

    # 6. Handle Debug/Subset Mismatch
    # predict_with_tta loads the full metadata file to get IDs, but if debug/max_samples
    # is active, the loader yields fewer batches. We truncate the IDs to match predictions.
    # Since the loader processes data in the order of the metadata file (head), this alignment is correct.
    if len(preds) != len(test_ids):
        logger.warning(
            f"Mismatch detected: {len(test_ids)} IDs vs {len(preds)} predictions. Truncating IDs to match predictions."
        )
        test_ids = test_ids[: len(preds)]

    # 7. Save Submission
    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": preds})

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {SUBMISSION_PATH}")
    logger.info(f"Submission shape: {submission_df.shape}")
