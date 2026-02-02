import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloader
from library.model import EEGNet


def predict(debug_subset_size=None):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        debug_subset_size (int, optional): If provided, limits the test set size for debugging.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    device = torch.device(Config.DEVICE)

    print(f"Inference device: {device}")

    # 2. Data Loader
    # mode="test" ensures no augmentation and no shuffling
    # load_cached_data=True allows using pre-processed .npy if available
    test_loader = get_dataloader(
        mode="test",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        debug_subset=debug_subset_size,
    )

    # 3. Model Initialization
    model = EEGNet(pretrained=False)  # Pretrained=False because we load custom weights
    model.to(device)

    # 4. Load Weights
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {Config.MODEL_PATH}. Train the model first."
        )

    print(f"Loading model weights from {Config.MODEL_PATH}...")
    checkpoint = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 5. Inference Loop
    all_preds = []

    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            # Move to device
            images = images.to(device, non_blocking=True)

            # Forward pass
            # Model output is already Softmax probabilities
            outputs = model(images)

            # Move to CPU and store
            all_preds.append(outputs.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_preds, axis=0)

    # 6. Prepare Submission DataFrame
    # Load test metadata to get eeg_ids
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug_subset_size is not None:
        test_df = test_df.head(debug_subset_size)

    # Ensure alignment
    if len(test_df) != len(predictions):
        raise ValueError(
            f"Mismatch between metadata rows ({len(test_df)}) and predictions ({len(predictions)})"
        )

    # Create DataFrame
    submission = pd.DataFrame(predictions, columns=Config.TARGET_COLS)

    # Add eeg_id
    submission.insert(0, "eeg_id", test_df["eeg_id"])

    # Rename columns to match submission format (prob -> vote)
    # Target cols in Config are like 'seizure_prob', submission needs 'seizure_vote'
    rename_map = {col: col.replace("_prob", "_vote") for col in Config.TARGET_COLS}
    submission.rename(columns=rename_map, inplace=True)

    # 7. Save Submission
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission.shape}")
    print("First 5 rows:")
    print(submission.head())

    return submission
