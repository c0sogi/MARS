import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloader
from library.model import AttentiveDualScaleNetwork


def predict(debug=False, load_cached_data=True):
    """
    Generates predictions for the test set using the trained AttentiveDualScaleNetwork.

    Args:
        debug (bool): If True, runs on a small subset of the test data.
        load_cached_data (bool): If True, attempts to load pre-processed .npy files for test data.

    Returns:
        pd.DataFrame: The submission dataframe containing eeg_id and predicted probabilities.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure output directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Logger
    log_file = os.path.join(Config.WORKING_DIR, "inference.log")
    logger = get_logger(log_file)
    logger.info("Starting Inference...")

    # 2. Load Metadata
    # We need the metadata to get the eeg_ids for the submission file
    test_df = pd.read_csv(Config.TEST_CSV)
    if debug:
        test_df = test_df.head(100)
        logger.info(f"Debug mode: limiting test set to {len(test_df)} samples.")

    # 3. Data Loader
    # Important: shuffle=False to maintain alignment with test_df eeg_ids
    logger.info("Initializing Test DataLoader...")
    test_loader = get_dataloader(
        mode="test",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=load_cached_data,
        shuffle=False,
    )

    # 4. Model Initialization
    logger.info("Loading Model...")
    model = AttentiveDualScaleNetwork()
    model.to(device)

    # Load weights
    if os.path.exists(Config.MODEL_PATH):
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {Config.MODEL_PATH}")
    else:
        logger.error(f"Model file not found at {Config.MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.eval()

    # 5. Inference Loop
    all_probs = []

    logger.info("Running prediction loop...")
    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):
            # If debug mode and we've exceeded the dataframe length (due to loader logic), break
            # Note: get_dataloader handles the subset logic internally if we passed a subset DF,
            # but here get_dataloader loads the full CSV inside library.data.
            # Ideally, library.data should handle debug slicing, but since we can't modify it,
            # we just run full inference or slice the output.
            # However, for strict debug compliance based on the prompt's constraint to not modify library files,
            # we will run the loader as provided. If debug is True, we just break early.

            if debug and batch_idx * Config.BATCH_SIZE >= len(test_df):
                break

            x_eeg, x_spec = inputs
            x_eeg = x_eeg.to(device)
            x_spec = x_spec.to(device)

            # Forward pass
            logits = model((x_eeg, x_spec))

            # Apply Softmax to get probabilities (sum to 1)
            probs = F.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_probs, axis=0)

    # Handle Debug Slicing if the loader returned more data than our debug metadata
    if len(predictions) > len(test_df):
        predictions = predictions[: len(test_df)]
    elif len(predictions) < len(test_df):
        # This might happen if debug=True caused early break
        test_df = test_df.iloc[: len(predictions)]

    # 6. Format Submission
    # The submission requires columns: eeg_id, [class]_vote
    # Config.TARGET_COLS are: seizure_prob, lpd_prob, etc.
    # We map them to: seizure_vote, lpd_vote, etc.

    submission_cols = ["eeg_id"] + [
        col.replace("_prob", "_vote") for col in Config.TARGET_COLS
    ]

    # Create DataFrame
    submission_df = pd.DataFrame(
        predictions,
        columns=[col.replace("_prob", "_vote") for col in Config.TARGET_COLS],
    )
    submission_df.insert(0, "eeg_id", test_df["eeg_id"].values)

    # 7. Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Submission shape: {submission_df.shape}")

    # Print first few rows for verification
    print(submission_df.head())

    return submission_df
