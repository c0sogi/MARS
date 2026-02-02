import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_logger, seed_everything
from library.data import BrainTumorDataset
from library.model import AsymmetricEfficientNet

logger = get_logger("predict")


def predict_submission():
    """
    Generates predictions for the test set using the trained model.
    Applies Test-Time Augmentation (TTA) and saves the result to submission.csv.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    logger.info("Starting inference/prediction phase...")

    # 2. Load Test Metadata
    if not os.path.exists(Config.TEST_CSV):
        logger.error(f"Test metadata not found at {Config.TEST_CSV}")
        return

    df_test = pd.read_csv(Config.TEST_CSV)
    logger.info(f"Loaded test metadata. Total samples: {len(df_test)}")

    # 3. Initialize Dataset and DataLoader
    # phase='test' ensures only ToTensorV2 is applied (no random augmentations)
    # load_cached_data=True enables the ROI caching mechanism defined in data.py
    test_dataset = BrainTumorDataset(df_test, phase="test", load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Must be False to maintain alignment with BraTS21ID
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Load Model
    model = AsymmetricEfficientNet()
    model = model.to(device)

    if not os.path.exists(Config.BEST_MODEL_PATH):
        logger.warning(
            f"Best model not found at {Config.BEST_MODEL_PATH}. Using random weights (debug mode)."
        )
    else:
        logger.info(f"Loading model weights from {Config.BEST_MODEL_PATH}")
        checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)

    model.eval()

    # 5. Inference Loop with TTA
    all_probs = []

    logger.info("Running inference with Test-Time Augmentation (TTA)...")

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            # --- TTA 1: Original ---
            logits_orig = model(inputs)
            probs_orig = torch.sigmoid(logits_orig)

            # --- TTA 2: Horizontal Flip ---
            # Input shape: (B, C, H, W). Flip on last dim (W).
            inputs_h = torch.flip(inputs, dims=[-1])
            logits_h = model(inputs_h)
            probs_h = torch.sigmoid(logits_h)

            # --- TTA 3: Vertical Flip ---
            # Input shape: (B, C, H, W). Flip on second to last dim (H).
            inputs_v = torch.flip(inputs, dims=[-2])
            logits_v = model(inputs_v)
            probs_v = torch.sigmoid(logits_v)

            # --- Average Predictions ---
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            # Flatten and store
            all_probs.extend(avg_probs.cpu().numpy().flatten())

    # 6. Generate Submission File
    if len(all_probs) != len(df_test):
        logger.error(
            f"Mismatch in prediction count: {len(all_probs)} vs {len(df_test)}"
        )

    df_test["MGMT_value"] = all_probs

    # Select required columns
    submission_df = df_test[["BraTS21ID", "MGMT_value"]]

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission file saved successfully to {Config.SUBMISSION_PATH}")

    # Print first few rows for verification
    print(submission_df.head())
