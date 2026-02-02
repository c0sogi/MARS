import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_loader import get_test_dataloader
from library.models import HybridEEGModel


def generate_submission(
    debug=Config.DEBUG,
    load_cached=True,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    model_path=Config.MODEL_PATH,
    output_path=Config.SUBMISSION_PATH,
):
    """
    Generates the submission file for the test set.

    Args:
        debug (bool): If True, runs on a subset of data.
        load_cached (bool): If True, attempts to load pre-processed data from disk.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for DataLoader.
        model_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("inference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting inference on device: {device}")

    # 2. Load Data
    # We load the test metadata directly to retrieve eeg_ids for the submission file
    # The DataLoader guarantees the order of samples matches the CSV because shuffle=False
    test_df = pd.read_csv(Config.TEST_CSV)
    if debug:
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)
        logger.info(f"Debug mode: Processing {len(test_df)} samples.")

    logger.info("Loading test data...")
    test_loader = get_test_dataloader(
        debug=debug,
        load_cached=load_cached,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # 3. Load Model
    logger.info(f"Loading model from {model_path}...")
    model = HybridEEGModel(
        num_classes=Config.N_CLASSES, pretrained_spec=False
    )  # Pretrained spec not needed for loading weights

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_preds = []

    logger.info("Running inference...")
    with torch.no_grad():
        for batch_idx, (raw_x, spec_x) in enumerate(test_loader):
            raw_x = raw_x.to(device)
            spec_x = spec_x.to(device)

            # Forward pass
            # Model outputs Softmax probabilities
            outputs = model(raw_x, spec_x)

            # Move to CPU and numpy
            preds = outputs.cpu().numpy()
            all_preds.append(preds)

    # Concatenate all batches
    if len(all_preds) > 0:
        final_preds = np.vstack(all_preds)
    else:
        # Handle empty case if necessary
        final_preds = np.zeros((0, Config.N_CLASSES))

    logger.info(f"Inference complete. Shape: {final_preds.shape}")

    # 5. Create Submission DataFrame
    # Ensure rows match
    if len(final_preds) != len(test_df):
        logger.error(
            f"Mismatch: {len(final_preds)} predictions vs {len(test_df)} metadata rows."
        )
        # In debug mode, this might happen if get_test_dataloader logic for debug differs slightly
        # from simple head(), but library code suggests consistency.

    submission_df = pd.DataFrame(final_preds, columns=Config.CLASS_NAMES)

    # Insert eeg_id at the beginning
    submission_df.insert(0, "eeg_id", test_df["eeg_id"].values)

    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")

    # Print first few rows for verification
    print(submission_df.head())
