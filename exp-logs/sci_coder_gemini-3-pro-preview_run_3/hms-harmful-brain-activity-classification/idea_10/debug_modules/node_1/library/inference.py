import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import EEGDataset
from library.model import MultiResDualStreamNet
from library.utils import get_logger, seed_everything


def predict(
    config=Config,
    checkpoint_path=None,
    output_path="./submission/submission.csv",
    batch_size=None,
    num_workers=None,
):
    """
    Runs inference on the test set and generates a submission file.

    Args:
        config (class): Configuration class containing parameters.
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to config.WORKING_DIR/best_model.pth.
        output_path (str): Path to save the submission CSV.
        batch_size (int, optional): Batch size for inference. Defaults to config.BATCH_SIZE.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to config.NUM_WORKERS.
    """
    # 1. Setup
    seed_everything(config.SEED)
    device = torch.device(config.DEVICE)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Setup Logger
    log_file = os.path.join(os.path.dirname(output_path), "inference.log")
    logger = get_logger(log_file)
    logger.info("Starting Inference Pipeline...")

    # Resolve defaults
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if batch_size is None:
        batch_size = config.BATCH_SIZE

    if num_workers is None:
        num_workers = config.NUM_WORKERS

    # 2. Load Data
    logger.info("Initializing Test Dataset...")
    # We use load_cached_data=True to leverage any pre-processed npy files
    # The dataset class handles generation if cache is missing.
    test_dataset = EEGDataset(mode="test", config=config, load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(f"Test Dataset loaded. Samples: {len(test_dataset)}")

    # 3. Load Model
    logger.info(f"Loading model from {checkpoint_path}...")
    model = MultiResDualStreamNet(
        pretrained=False
    )  # Pretrained weights not needed for loading state_dict

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_preds = []

    logger.info("Running inference...")
    with torch.no_grad():
        for step, ((x_a, x_b), _) in enumerate(test_loader):
            # Move inputs to device
            x_a = x_a.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)

            # Forward pass (returns logits)
            logits = model((x_a, x_b))

            # Apply Softmax to get probabilities
            probs = F.softmax(logits, dim=1)

            # Move to CPU and store
            all_preds.append(probs.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)

    # 5. Generate Submission
    logger.info("Generating submission file...")

    # Load test metadata to get eeg_ids
    test_df = pd.read_csv(config.TEST_CSV)

    # Verify alignment
    if len(test_df) != len(all_preds):
        raise ValueError(
            f"Mismatch: Metadata has {len(test_df)} rows, Predictions have {len(all_preds)} rows."
        )

    # Create Submission DataFrame
    submission = pd.DataFrame()
    submission["eeg_id"] = test_df["eeg_id"]

    # Assign probabilities to the correct columns
    # Config.CLASS_NAMES contains ['seizure_vote', 'lpd_vote', ...]
    for i, col_name in enumerate(config.CLASS_NAMES):
        submission[col_name] = all_preds[:, i]

    # Save to CSV
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")

    # Print head for verification
    print(submission.head())

    return submission
