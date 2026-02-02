import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger, set_seed
from library.model import build_model
from library.data import get_dataloaders


def predict_and_submit(load_cached_data=True):
    """
    Runs the inference pipeline on the test set and generates the submission file.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed test data
                                 from the cache defined in Config.
    """
    # 1. Setup
    logger = setup_logger("Inference")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    logger.info(f"Starting inference on device: {device}")

    # 2. Load Data
    # We only need the test loader and the test_ids.
    # The data module handles the caching and 3-instance expansion logic.
    _, _, test_loader, test_ids = get_dataloaders(load_cached_data=load_cached_data)

    logger.info(f"Test data loaded. Total instances: {len(test_ids)}")

    # 3. Load Model
    model = build_model(device=device)

    # Path to the best model saved during training
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Model loaded from {model_path}")

    # 4. Inference Loop
    all_probs = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)

            # Apply sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits)

            # Move to CPU and flatten to 1D array
            all_probs.append(probs.cpu().numpy().flatten())

    # Concatenate all batch results
    all_probs = np.concatenate(all_probs)

    # 5. Aggregation
    # We have 3 instances per subject. We need to average them.
    # test_ids contains the BraTS21ID for each instance.

    df_pred = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": all_probs})

    # Group by Subject ID and calculate the mean probability (Consensus Aggregation)
    df_submission = df_pred.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()

    # 6. Formatting and Saving
    # Ensure submission directory exists
    submission_dir = os.path.dirname(Config.SUBMISSION_PATH)
    os.makedirs(submission_dir, exist_ok=True)

    # Save to CSV
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Total subjects predicted: {len(df_submission)}")

    # Print first few rows for verification
    logger.info("First 5 predictions:")
    logger.info(f"\n{df_submission.head()}")
